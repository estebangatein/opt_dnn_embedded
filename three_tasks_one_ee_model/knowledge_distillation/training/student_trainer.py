from one_ee_ptq_model import OneEE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from datetime import datetime
from onnxruntime.quantization import quantize_static, QuantType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = OneEE()
teacher.load_state_dict(torch.load('teacher_trained.pth'))
teacher.eval()

class StudentOneEE(nn.Module):
    def __init__(self):
        super(StudentOneEE, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # early exit for object detection
        self.obj_detect_fc_ee = nn.Linear(32 * 50 * 50, 4)

        # additional pooling layer to reduce dimensions
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.obj_detect_fc_final = nn.Linear(128 * 6 * 6, 4)

        self.binary_fc1 = nn.Linear(128 * 6 * 6, 64)
        self.binary_fc2 = nn.Linear(64, 1)

        self.regression_fc1 = nn.Linear(128 * 6 * 6, 64)
        self.regression_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        obj_detect_ee = x.view(-1, 32 * 50 * 50)
        obj_detect_ee = self.obj_detect_fc_ee(obj_detect_ee)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.pool5(x)

        x = x.flatten(1)

        obj_detect_final = self.obj_detect_fc_final(x)

        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return obj_detect_ee, obj_detect_final, binary_output, regression_output

student = StudentOneEE()

criterion_classification = nn.CrossEntropyLoss()  
criterion_binary = nn.BCEWithLogitsLoss()  
criterion_regression = nn.MSELoss()  
optimizer = optim.Adam(student.parameters(), lr=0.001)



# define KL divergence loss
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_teacher_probs = torch.softmax(teacher_logits / temperature, dim=1)
    soft_student_probs = torch.softmax(student_logits / temperature, dim=1)

    # use of KLDivLoss between softened probs
    # loss = nn.KLDivLoss(reduction='batchmean')(torch.log(soft_student_probs), soft_teacher_probs)
    epsilon = 1e-10
    kl_loss = torch.sum(soft_student_probs * torch.log((soft_student_probs + epsilon) / (soft_teacher_probs + epsilon)))


    return kl_loss * (temperature ** 2)

def train_student_model(
    train_dataloader, val_dataloader, student_model, teacher_model, optimizer,
    criterion_classification, criterion_regression, criterion_binary, epochs=10, temperature=2.0
):
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # don't update teacher model

    metrics = {
        "train": {
            "losses": [],
            "accuracies": {"obj_detect_ee": [], "obj_detect": [], "binary": []},
            "mse": []
        },
        "val": {
            "losses": [],
            "accuracies": {"obj_detect_ee": [], "obj_detect": [], "binary": []},
            "mse": []
        }
    }

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        
        student_model.train()
        train_losses = {"obj_detect_ee": 0.0, "obj_detect": 0.0, "binary": 0.0, "regression": 0.0, "distillation": 0.0}
        train_counts = {key: 0 for key in train_losses.keys()}
        correct_ee, correct_final, correct_binary = 0, 0, 0
        mse_train = 0.0
        total_loss_train = 0.0

        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            
            # student forward pass
            student_pred_ee, student_pred_final, student_binary_pred, student_regression_pred = student_model(images)
            student_regression_pred = student_regression_pred.squeeze(-1)
            # teacher forward pass
            with torch.no_grad():
                teacher_pred_ee, teacher_pred_final, teacher_binary_pred, teacher_regression_pred = teacher_model(images)
                teacher_regression_pred = teacher_regression_pred.squeeze(-1)
            
            obj_detect_labels = torch.stack([label['obj_detect'].clone().detach() for label in labels]).to(device)
            binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
            regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)
            
            obj_detect_mask = obj_detect_labels.sum(dim=1) > 0
            binary_mask = binary_labels != -9999
            regression_mask = regression_labels != -9999.0

            total_loss_batch = torch.tensor(0.0, device=device)
            
            # KL loss
            distillation_loss_value = 0.0
            if obj_detect_mask.any():
                distillation_loss_value += distillation_loss(student_pred_ee[obj_detect_mask], teacher_pred_ee[obj_detect_mask], temperature)
                distillation_loss_value += distillation_loss(student_pred_final[obj_detect_mask], teacher_pred_final[obj_detect_mask], temperature)
            
            train_losses["distillation"] += distillation_loss_value
            total_loss_batch += distillation_loss_value
            # hard loss
            if obj_detect_mask.any():
                loss_obj_detect_ee = criterion_classification(
                    student_pred_ee[obj_detect_mask].float(),
                    obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                )
                train_losses["obj_detect_ee"] += loss_obj_detect_ee.item()
                train_counts["obj_detect_ee"] += obj_detect_mask.sum().item()
                total_loss_batch += loss_obj_detect_ee

                loss_obj_detect = criterion_classification(
                    student_pred_final[obj_detect_mask].float(),
                    obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                )
                train_losses["obj_detect"] += loss_obj_detect.item()
                train_counts["obj_detect"] += obj_detect_mask.sum().item()
                total_loss_batch += loss_obj_detect

            if binary_mask.any():
                loss_binary = criterion_binary(
                    student_binary_pred[binary_mask].float(),  
                    binary_labels[binary_mask].float().unsqueeze(-1)  
                )
                train_losses["binary"] += loss_binary.item()
                train_counts["binary"] += binary_mask.sum().item()
                total_loss_batch += loss_binary

            if regression_mask.any():
                loss_regression = criterion_regression(
                    student_regression_pred[regression_mask].float(),
                    regression_labels[regression_mask].float()
                )
                train_losses["regression"] += loss_regression.item()
                train_counts["regression"] += regression_mask.sum().item()
                total_loss_batch += loss_regression

            total_loss_batch.backward()
            optimizer.step()
            total_loss_train += total_loss_batch.item()

        avg_loss_train = total_loss_train / max(sum(train_counts.values()), 1)
        metrics["train"]["losses"].append(avg_loss_train)
        metrics["train"]["accuracies"]["obj_detect_ee"].append(correct_ee / max(train_counts["obj_detect_ee"], 1))
        metrics["train"]["accuracies"]["obj_detect"].append(correct_final / max(train_counts["obj_detect"], 1))
        metrics["train"]["accuracies"]["binary"].append(correct_binary / max(train_counts["binary"], 1))
        metrics["train"]["mse"].append(mse_train / max(train_counts["regression"], 1))

        print(f"Training Loss: {avg_loss_train:.4f}")
        print(f"Training Loss (obj_detect_ee): {train_losses['obj_detect_ee'] / max(train_counts['obj_detect_ee'], 1):.4f}")
        print(f"Training Loss (obj_detect): {train_losses['obj_detect'] / max(train_counts['obj_detect'], 1):.4f}")
        print(f"Training Loss (binary): {train_losses['binary'] / max(train_counts['binary'], 1):.4f}")
        print(f"Training Loss (regression): {train_losses['regression'] / max(train_counts['regression'], 1):.4f}")
        print(f"Training Distillation Loss: {train_losses['distillation'] / max(train_counts['obj_detect_ee'], 1):.4f}")
        print(f"Training Accuracy (obj_detect_ee): {metrics['train']['accuracies']['obj_detect_ee'][-1]:.4f}")
        print(f"Training Accuracy (obj_detect): {metrics['train']['accuracies']['obj_detect'][-1]:.4f}")
        print(f"Training Accuracy (binary): {metrics['train']['accuracies']['binary'][-1]:.4f}")
        print(f"Training MSE: {metrics['train']['mse'][-1]:.4f}")
        
        # same for validation
        student_model.eval()  
        val_losses = {key: 0.0 for key in train_losses.keys()} 
        val_counts = {key: 0 for key in train_losses.keys()} 
        correct_ee_val, correct_final_val, correct_binary_val = 0, 0, 0
        mse_val = 0.0
        total_loss_val = 0.0

        with torch.no_grad(): 
            for images, labels in tqdm(val_dataloader): 
                images = images.to(device)

                student_pred_ee, student_pred_final, student_binary_pred, student_regression_pred = student_model(images)
                student_regression_pred = student_regression_pred.squeeze(-1) 
            
                teacher_pred_ee, teacher_pred_final, teacher_binary_pred, teacher_regression_pred = teacher_model(images)
                teacher_regression_pred = teacher_regression_pred.squeeze(-1)
            
                obj_detect_labels = torch.stack([label['obj_detect'].clone().detach() for label in labels]).to(device)
                binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
                regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)
            
                obj_detect_mask = obj_detect_labels.sum(dim=1) > 0 
                binary_mask = binary_labels != -9999 
                regression_mask = regression_labels != -9999.0 
            
                total_loss_batch = torch.tensor(0.0, device=device) 
            
                distillation_loss_value = 0.0
                if obj_detect_mask.any():
                    distillation_loss_value += distillation_loss(student_pred_ee[obj_detect_mask], teacher_pred_ee[obj_detect_mask], temperature)
                    distillation_loss_value += distillation_loss(student_pred_final[obj_detect_mask], teacher_pred_final[obj_detect_mask], temperature)

                val_losses["distillation"] += distillation_loss_value
                total_loss_batch += distillation_loss_value

                if obj_detect_mask.any():
                    loss_obj_detect_ee = criterion_classification(
                        student_pred_ee[obj_detect_mask].float(),
                        obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                    )
                    val_losses["obj_detect_ee"] += loss_obj_detect_ee.item()
                    val_counts["obj_detect_ee"] += obj_detect_mask.sum().item()
                    total_loss_batch += loss_obj_detect_ee

                    loss_obj_detect = criterion_classification(
                        student_pred_final[obj_detect_mask].float(),
                        obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                    )
                    val_losses["obj_detect"] += loss_obj_detect.item()
                    val_counts["obj_detect"] += obj_detect_mask.sum().item()
                    total_loss_batch += loss_obj_detect

                if binary_mask.any():
                    loss_binary = criterion_binary(
                        student_binary_pred[binary_mask].float(),
                        binary_labels[binary_mask].float().unsqueeze(-1)
                    )
                    val_losses["binary"] += loss_binary.item()
                    val_counts["binary"] += binary_mask.sum().item()
                    total_loss_batch += loss_binary

                if regression_mask.any():
                    loss_regression = criterion_regression(
                        student_regression_pred[regression_mask].float(),
                        regression_labels[regression_mask].float()
                    )
                    val_losses["regression"] += loss_regression.item()
                    val_counts["regression"] += regression_mask.sum().item()
                    total_loss_batch += loss_regression

                total_loss_val += total_loss_batch.item()

        avg_loss_val = total_loss_val / max(sum(val_counts.values()), 1)
        metrics["val"]["losses"].append(avg_loss_val)

        print(f"Validation Loss: {avg_loss_val:.4f}")
        print(f"Validation Distillation Loss: {val_losses['distillation'] / max(val_counts['obj_detect_ee'], 1):.4f}")


    return student_model, metrics





class MultiTaskDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)  
        label = self.labels[idx]
        return image, label

def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), labels

def dataset_creator(path_to_folder, max_by_experiment=3):
    col, fire, steer = 0, 0, 0
    images, labels = [], []

    for folder in os.listdir(path_to_folder):
        folder_path = os.path.join(path_to_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                labels_txt_path = os.path.join(folder_path, file)

                if file.endswith('.txt') and 'labels' in file and col < max_by_experiment:
                    col += 1
                    img_list = []
                    img_dir = os.path.join(folder_path, 'images')
                    
                    for pic in sorted(os.listdir(img_dir)):
                        img = load_img(os.path.join(img_dir, pic), target_size=(200, 200), color_mode='grayscale')
                        img_array = (img_to_array(img) / 128.0) - 1.0  
                        img_list.append(img_array)
                    
                    labels_txt = np.loadtxt(labels_txt_path)
                    for img_array, label_val in zip(img_list, labels_txt):
                        images.append(img_array)
                        label = {
                            "obj_detect": torch.zeros(4),  
                            "binary": torch.tensor(label_val) if label_val in [0, 1] else -9999.0,  
                            "regression": -9999.0  
                        }
                        labels.append(label)

                elif file.endswith('.txt') and 'fire' in file and fire < max_by_experiment:
                    fire += 1
                    img_list = []
                    img_dir = os.path.join(folder_path, 'images')
                    for pic in sorted(os.listdir(img_dir)):
                        img = load_img(os.path.join(img_dir, pic), target_size=(200, 200), color_mode='grayscale')
                        img_array = (img_to_array(img) / 128.0) - 1.0  
                        img_list.append(img_array)

                    labels_txt = np.loadtxt(labels_txt_path, delimiter=' ')
                    for img_array, label_vals in zip(img_list, labels_txt):
                        images.append(img_array)
                        if label_vals.size == 4:
                            obj_detect_tensor = torch.tensor(label_vals, dtype=torch.float32)
                        else:
                            obj_detect_tensor = torch.zeros(4)  
                        label = {
                            "obj_detect": obj_detect_tensor,  
                            "binary": -9999.0,  
                            "regression": -9999.0  
                        }
                        labels.append(label)

                elif file.endswith('.txt') and 'sync' in file and steer < max_by_experiment:
                    steer += 1
                    img_list = []
                    img_dir = os.path.join(folder_path, 'images')
                    for pic in sorted(os.listdir(img_dir)):
                        img = load_img(os.path.join(img_dir, pic), target_size=(200, 200), color_mode='grayscale')
                        img_array = (img_to_array(img) / 128.0) - 1.0  
                        img_list.append(img_array)

                    labels_txt = np.loadtxt(labels_txt_path, usecols=0, delimiter=',', skiprows=1)
                    for img_array, label_val in zip(img_list, labels_txt):
                        images.append(img_array)
                        label = {
                            "obj_detect": torch.zeros(4),  
                            "binary": -9999.0,  
                            "regression": torch.tensor(label_val, dtype=torch.float32) if not np.isnan(label_val) else -9999.0  
                        }
                        labels.append(label)

    return images, labels

images_train, labels_train = dataset_creator('../data/merged_data/training', 50)
images_val, labels_val = dataset_creator('../data/merged_data/validation', 20)

indices = np.random.permutation(len(images_train))
images_train = [images_train[i] for i in indices]
labels_train = [labels_train[i] for i in indices]

indices = np.random.permutation(len(images_val))
images_val = [images_val[i] for i in indices]
labels_val = [labels_val[i] for i in indices]

train_dataset = MultiTaskDataset(images=images_train, labels=labels_train)
val_dataset = MultiTaskDataset(images=images_val, labels=labels_val)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

trained_model, loss_data = train_student_model(train_dataloader, val_dataloader, student, teacher, optimizer, criterion_classification, criterion_regression, criterion_binary, epochs=10)

trained_model.to('cpu')
torch.save(trained_model.state_dict(), 'trained_model.pth')

with open('training_metrics.pkl', "wb") as f:
        pickle.dump(loss_data, f)

dummy_input = torch.randn(1, 1, 200, 200)
torch.onnx.export(trained_model, dummy_input, "trained_model.onnx", input_names=['input'], output_names=['obj_detect_ee', 'obj_detect_final', 'binary_output', 'regression_output'])


class CalibrationDataReader:
    def __init__(self, calibration_dir):
        self.calibration_files = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.endswith('.npy')]
        self.data_index = 0
    
    def get_next(self):
        if self.data_index < len(self.calibration_files):
            input_data = np.load(self.calibration_files[self.data_index])
            input_data = np.expand_dims(input_data, axis=0)  
            self.data_index += 1
            return {'input': input_data.astype(np.float32)}
        else:
            return None

    def rewind(self):
        self.data_index = 0


calibration_dir = './calibration_data'
calibration_data_reader = CalibrationDataReader(calibration_dir)


onnx_model_path = 'trained_model.onnx'
onnx_quantized_model_path = 'quantized_model.onnx'

quantize_static(
    onnx_model_path,                      
    onnx_quantized_model_path,            
    calibration_data_reader,              
    quant_format=QuantType.QInt8,         
    weight_type=QuantType.QInt8           
)

print(f"Quantized model saved to {onnx_quantized_model_path}")


print(f'Script finalizado en fecha y hora: {datetime.now()}')
