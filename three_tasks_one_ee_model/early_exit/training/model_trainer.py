
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
from tensorflow.keras.utils import load_img, img_to_array
from tqdm import tqdm
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OneEE(nn.Module):
    def __init__(self):
        super(OneEE, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.obj_detect_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.obj_detect_bn = nn.BatchNorm2d(128)
        self.obj_detect_fc_ee = nn.Linear(128 * 50 * 50, 4)  

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.obj_detect_fc_final = nn.Linear(512 * 25 * 25, 4)

        self.binary_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.binary_fc2 = nn.Linear(256, 1)

        self.regression_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.regression_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # ee for object detection
        obj_detect_ee = F.relu(self.obj_detect_bn(self.obj_detect_conv(x)))
        obj_detect_ee = torch.flatten(obj_detect_ee, 1)  
        obj_detect_ee = self.obj_detect_fc_ee(obj_detect_ee)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  

        obj_detect_final = self.obj_detect_fc_final(x)

        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return obj_detect_ee, obj_detect_final, binary_output, regression_output


model = OneEE()
criterion_classification = nn.CrossEntropyLoss()  
criterion_binary = nn.BCEWithLogitsLoss()  
criterion_regression = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(
    train_dataloader, val_dataloader, model, optimizer,
    criterion_classification, criterion_regression, criterion_binary,epochs=10
):
    model = model.to(device)
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
        
        model.train()
        train_losses = {"obj_detect_ee": 0.0, "obj_detect": 0.0, "binary": 0.0, "regression": 0.0}
        train_counts = {key: 0 for key in train_losses.keys()}
        correct_ee, correct_final, correct_binary = 0, 0, 0
        mse_train = 0.0
        total_loss_train = 0.0

        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            
            obj_detect_pred_ee, obj_detect_pred_final, binary_pred, regression_pred = model(images)
            regression_pred = regression_pred.squeeze(-1)
            
            obj_detect_labels = torch.stack([label['obj_detect'].clone().detach() for label in labels]).to(device)
            binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
            regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)
            
            obj_detect_mask = obj_detect_labels.sum(dim=1) > 0
            binary_mask = binary_labels != -9999
            regression_mask = regression_labels != -9999.0

            total_loss_batch = torch.tensor(0.0, device=device)
            
            if obj_detect_mask.any():
                loss_obj_detect_ee = criterion_classification(
                    obj_detect_pred_ee[obj_detect_mask].float(),
                    obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                )
                train_losses["obj_detect_ee"] += loss_obj_detect_ee.item()
                train_counts["obj_detect_ee"] += obj_detect_mask.sum().item()
                total_loss_batch += loss_obj_detect_ee

                correct_ee += (obj_detect_pred_ee[obj_detect_mask].argmax(dim=1) == obj_detect_labels[obj_detect_mask].argmax(dim=1)).sum().item()

                loss_obj_detect = criterion_classification(
                    obj_detect_pred_final[obj_detect_mask].float(),
                    obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                )
                train_losses["obj_detect"] += loss_obj_detect.item()
                train_counts["obj_detect"] += obj_detect_mask.sum().item()
                total_loss_batch += loss_obj_detect

                correct_final += (obj_detect_pred_final[obj_detect_mask].argmax(dim=1) == obj_detect_labels[obj_detect_mask].argmax(dim=1)).sum().item()
                

            if binary_mask.any():
                # binary loss (BCEWithLogitsLoss includes sigmoid)
                loss_binary = criterion_binary(
                    binary_pred[binary_mask].float(),  
                    binary_labels[binary_mask].float().unsqueeze(-1)  
                )
                train_losses["binary"] += loss_binary.item()
                train_counts["binary"] += binary_mask.sum().item()
                total_loss_batch += loss_binary

                probabilities = torch.sigmoid(binary_pred[binary_mask].float())
                binary_predictions = (probabilities > 0.5).float()
                correct_binary += (binary_predictions == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()



            if regression_mask.any():
                loss_regression = criterion_regression(
                    regression_pred[regression_mask].float(),
                    regression_labels[regression_mask].float()
                )
                train_losses["regression"] += loss_regression.item()
                train_counts["regression"] += regression_mask.sum().item()
                total_loss_batch += loss_regression

                mse_train += loss_regression.item() * regression_mask.sum().item()

            total_loss_batch.backward()
            optimizer.step()
            total_loss_train += total_loss_batch.item()

        # save the metrics
        avg_loss_train = total_loss_train / max(sum(train_counts.values()), 1)
        metrics["train"]["losses"].append(avg_loss_train)
        metrics["train"]["accuracies"]["obj_detect_ee"].append(correct_ee / max(train_counts["obj_detect_ee"], 1))
        metrics["train"]["accuracies"]["obj_detect"].append(correct_final / max(train_counts["obj_detect"], 1))
        metrics["train"]["accuracies"]["binary"].append(correct_binary / max(train_counts["binary"], 1))
        metrics["train"]["mse"].append(mse_train / max(train_counts["regression"], 1))

        # print for monitoring
        print(f"Training Loss: {avg_loss_train:.4f}")
        print(f"Training Loss (obj_detect_ee): {train_losses['obj_detect_ee'] / max(train_counts['obj_detect_ee'], 1):.4f}")
        print(f"Training Loss (obj_detect): {train_losses['obj_detect'] / max(train_counts['obj_detect'], 1):.4f}")
        print(f"Training Loss (binary): {train_losses['binary'] / max(train_counts['binary'], 1):.4f}")
        print(f"Training Loss (regression): {train_losses['regression'] / max(train_counts['regression'], 1):.4f}")
        print(f"Training Accuracy (obj_detect_ee): {metrics['train']['accuracies']['obj_detect_ee'][-1]:.4f}")
        print(f"Training Accuracy (obj_detect): {metrics['train']['accuracies']['obj_detect'][-1]:.4f}")
        print(f"Training Accuracy (binary): {metrics['train']['accuracies']['binary'][-1]:.4f}")
        print(f"Training MSE: {metrics['train']['mse'][-1]:.4f}")

        # validation
        model.eval()
        val_losses = {key: 0.0 for key in train_losses.keys()}
        val_counts = {key: 0 for key in train_losses.keys()}
        correct_ee_val, correct_final_val, correct_binary_val = 0, 0, 0
        mse_val = 0.0
        total_loss_val = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images = images.to(device)

                obj_detect_pred_ee, obj_detect_pred_final, binary_pred, regression_pred = model(images)
                regression_pred = regression_pred.squeeze(-1)
                
                obj_detect_labels = torch.stack([label['obj_detect'].clone().detach() for label in labels]).to(device)
                binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
                regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)
                
                obj_detect_mask = obj_detect_labels.sum(dim=1) > 0
                binary_mask = binary_labels != -9999
                regression_mask = regression_labels != -9999.0

                total_loss_batch = torch.tensor(0.0, device=device)

                if obj_detect_mask.any():
                    loss_obj_detect_ee = criterion_classification(
                        obj_detect_pred_ee[obj_detect_mask].float(),
                        obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                    )
                    val_losses["obj_detect_ee"] += loss_obj_detect_ee.item()
                    val_counts["obj_detect_ee"] += obj_detect_mask.sum().item()
                    total_loss_batch += loss_obj_detect_ee

                    correct_ee_val += (obj_detect_pred_ee[obj_detect_mask].argmax(dim=1) == obj_detect_labels[obj_detect_mask].argmax(dim=1)).sum().item()

                    loss_obj_detect = criterion_classification(
                        obj_detect_pred_final[obj_detect_mask].float(),
                        obj_detect_labels[obj_detect_mask].argmax(dim=1).long()
                    )
                    val_losses["obj_detect"] += loss_obj_detect.item()
                    val_counts["obj_detect"] += obj_detect_mask.sum().item()
                    total_loss_batch += loss_obj_detect

                    correct_final_val += (obj_detect_pred_final[obj_detect_mask].argmax(dim=1) == obj_detect_labels[obj_detect_mask].argmax(dim=1)).sum().item()

                if binary_mask.any():
                    loss_binary = criterion_binary(
                        binary_pred[binary_mask].float(),
                        binary_labels[binary_mask].float().unsqueeze(-1)
                    )
                    val_losses["binary"] += loss_binary.item()
                    val_counts["binary"] += binary_mask.sum().item()
                    total_loss_batch += loss_binary

                    correct_binary_val += ((binary_pred[binary_mask] > 0.5).float() == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()


                if regression_mask.any():
                    loss_regression = criterion_regression(
                        regression_pred[regression_mask].float(),
                        regression_labels[regression_mask].float()
                    )
                    val_losses["regression"] += loss_regression.item()
                    val_counts["regression"] += regression_mask.sum().item()
                    total_loss_batch += loss_regression

                    mse_val += loss_regression.item() * regression_mask.sum().item()

                total_loss_val += total_loss_batch.item()

        avg_loss_val = total_loss_val / max(sum(val_counts.values()), 1)
        metrics["val"]["losses"].append(avg_loss_val)
        metrics["val"]["accuracies"]["obj_detect_ee"].append(correct_ee_val / max(val_counts["obj_detect_ee"], 1))
        metrics["val"]["accuracies"]["obj_detect"].append(correct_final_val / max(val_counts["obj_detect"], 1))
        metrics["val"]["accuracies"]["binary"].append(correct_binary_val / max(val_counts["binary"], 1))
        metrics["val"]["mse"].append(mse_val / max(val_counts["regression"], 1))

        print(f"Validation Loss: {avg_loss_val:.4f}")
        print(f"Validation Loss (obj_detect_ee): {val_losses['obj_detect_ee'] / max(val_counts['obj_detect_ee'], 1):.4f}")
        print(f"Validation Loss (obj_detect): {val_losses['obj_detect'] / max(val_counts['obj_detect'], 1):.4f}")
        print(f"Validation Loss (binary): {val_losses['binary'] / max(val_counts['binary'], 1):.4f}")
        print(f"Validation Loss (regression): {val_losses['regression'] / max(val_counts['regression'], 1):.4f}")
        print(f"Validation Accuracy (obj_detect_ee): {metrics['val']['accuracies']['obj_detect_ee'][-1]:.4f}")
        print(f"Validation Accuracy (obj_detect): {metrics['val']['accuracies']['obj_detect'][-1]:.4f}")
        print(f"Validation Accuracy (binary): {metrics['val']['accuracies']['binary'][-1]:.4f}")
        print(f"Validation MSE: {metrics['val']['mse'][-1]:.4f}")

    # save the metrics
    with open("training_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    return model, metrics



class MultiTaskDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)  # change to CHW format
        label = self.labels[idx]
        return image, label

# keeps None in labels
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

images_train, labels_train = dataset_creator('../data/merged_data/training', 100)
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



trained_model, _ = train_model(train_dataloader, val_dataloader, model,optimizer, criterion_classification, criterion_regression, criterion_binary, epochs=50)


trained_model = trained_model.to('cpu')
trained_model.eval()

torch.save(trained_model.state_dict(), 'trained_model.pth')

dummy_input = torch.randn(1, 1, 200, 200)
onnx_model_path = 'model_trained.onnx'
torch.onnx.export(trained_model, dummy_input, onnx_model_path, input_names=['input'], output_names=['early_exit', 'final_exit'])

print(f'Script finalizado en fecha y hora: {datetime.now()}')



