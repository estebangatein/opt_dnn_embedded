
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
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoEE(nn.Module):
    def __init__(self):
        super(TwoEE, self).__init__()

        # Bloque inicial de capas convolucionales para extracción de características
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Más capas convolucionales y pooling para incrementar complejidad
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ee_branch_binary = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ee_branch_binary_bn = nn.BatchNorm2d(128)
        self.ee_branch_binary_fc = nn.Linear(128 * 50 * 50, 1)

        self.ee_branch_reg = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ee_branch_reg_bn = nn.BatchNorm2d(128)
        self.ee_branch_reg_fc = nn.Linear(128 * 50 * 50, 1)

        # Capas adicionales para el resto de tareas
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Clasificador binario
        self.binary_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.binary_fc2 = nn.Linear(256, 1)

        # Salida de regresión
        self.regression_fc1 = nn.Linear(512 * 25 * 25, 256)
        self.regression_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Primer bloque convolucional
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Segundo bloque convolucional
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Rama para clasificación binaria
        ee_branch_binary = F.relu(self.ee_branch_binary_bn(self.ee_branch_binary(x)))
        ee_branch_binary = torch.flatten(ee_branch_binary, 1)
        ee_branch_binary = self.ee_branch_binary_fc(ee_branch_binary)

        # Rama para regresión
        ee_branch_reg = F.relu(self.ee_branch_reg_bn(self.ee_branch_reg(x)))
        ee_branch_reg = torch.flatten(ee_branch_reg, 1)
        ee_branch_reg = self.ee_branch_reg_fc(ee_branch_reg)

        # Continuamos con el resto de capas para las otras tareas
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  # Aplanar

        # Clasificación binaria
        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        # Regresión
        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return ee_branch_binary, ee_branch_reg, binary_output, regression_output


model = TwoEE()
criterion_binary = nn.BCEWithLogitsLoss()  # Para la salida binaria
criterion_regression = nn.MSELoss()  # Para la salida de regresión
criterion_regression = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train_model(
    train_dataloader, val_dataloader, model, optimizer, criterion_regression, criterion_binary, epochs=10):
    model = model.to(device)
    metrics = {
        "train": {
            "losses": [],
            "accuracies": {"binary_ee": [], "binary": []},
            "mse": {"ee": [], "regression": []}
        },
        "val": {
            "losses": [],
            "accuracies": {"binary_ee": [], "binary": []},
            "mse": {"ee": [], "regression": []}
        }
    }

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        
        # Fase de entrenamiento
        model.train()
        train_losses = {"binary_ee": 0.0, "regression_ee": 0.0, "binary": 0.0, "regression": 0.0}
        train_counts = {key: 0 for key in train_losses.keys()}
        correct_binary_ee, correct_binary = 0, 0
        mse_train_ee, mse_train = 0.0, 0.0
        total_loss_train = 0.0

        for images, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            
            # Forward pass en el conjunto de entrenamiento
            binary_pred_ee, regression_pred_ee, binary_pred, regression_pred = model(images)
            regression_pred = regression_pred.squeeze(-1)
            regression_pred_ee = regression_pred_ee.squeeze(-1)
            
            # Convertir etiquetas a tensores y enviarlas al dispositivo
            binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
            regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)
            
            # Enmascaramiento
            binary_mask = binary_labels != -9999
            regression_mask = regression_labels != -9999.0

            # Calcular pérdidas
            total_loss_batch = torch.tensor(0.0, device=device)
            

            if binary_mask.any():
                # Calcular la pérdida de la tarea binaria (BCEWithLogitsLoss ya incluye la sigmoide implícita)
                loss_binary = criterion_binary(
                    binary_pred[binary_mask].float(),  # Predicciones del modelo
                    binary_labels[binary_mask].float().unsqueeze(-1)  # Etiquetas binarias
                )
                train_losses["binary"] += loss_binary.item()
                train_counts["binary"] += binary_mask.sum().item()
                total_loss_batch += loss_binary

                probabilities = torch.sigmoid(binary_pred[binary_mask].float())
                binary_predictions = (probabilities > 0.5).float()
                correct_binary += (binary_predictions == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()

                loss_binary_ee = criterion_binary(
                    binary_pred_ee[binary_mask].float(),
                    binary_labels[binary_mask].float().unsqueeze(-1)
                )

                train_losses["binary_ee"] += loss_binary_ee.item()
                train_counts["binary_ee"] += binary_mask.sum().item()
                total_loss_batch += loss_binary_ee

                probabilities_ee = torch.sigmoid(binary_pred_ee[binary_mask].float())
                binary_predictions_ee = (probabilities_ee > 0.5).float()
                correct_binary_ee += (binary_predictions_ee == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()



            if regression_mask.any():
                loss_regression = criterion_regression(
                    regression_pred[regression_mask].float(),
                    regression_labels[regression_mask].float()
                )
                train_losses["regression"] += loss_regression.item()
                train_counts["regression"] += regression_mask.sum().item()
                total_loss_batch += loss_regression

                # Calcular MSE para regresión
                mse_train += 3 * loss_regression.item() * regression_mask.sum().item()

                loss_regression_ee = criterion_regression(
                    regression_pred_ee[regression_mask].float(),
                    regression_labels[regression_mask].float()
                )
                train_losses["regression_ee"] += loss_regression_ee.item()
                train_counts["regression_ee"] += regression_mask.sum().item()
                total_loss_batch += loss_regression_ee

                # Calcular MSE para regresión_ee
                mse_train_ee += loss_regression_ee.item() * regression_mask.sum().item()


            # Backpropagación
            total_loss_batch.backward()
            optimizer.step()
            total_loss_train += total_loss_batch.item()

        # Guardar métricas de entrenamiento
        avg_loss_train = total_loss_train / max(sum(train_counts.values()), 1)
        metrics["train"]["losses"].append(avg_loss_train)
        metrics["train"]["accuracies"]["binary_ee"].append(correct_binary_ee / max(train_counts["binary_ee"], 1))
        metrics["train"]["accuracies"]["binary"].append(correct_binary / max(train_counts["binary"], 1))
        metrics["train"]["mse"]["regression"].append(mse_train / max(train_counts["regression"], 1))
        metrics["train"]["mse"]["ee"].append(mse_train_ee / max(train_counts["regression_ee"], 1))

        # Imprimir métricas de entrenamiento
        print(f"Training Loss: {avg_loss_train:.4f}")
        print(f"Training Loss (binary_ee): {train_losses['binary_ee'] / max(train_counts['binary_ee'], 1):.4f}")
        print(f"Training Loss (binary): {train_losses['binary'] / max(train_counts['binary'], 1):.4f}")
        print(f'Training Loss (regression_ee): {train_losses["regression_ee"] / max(train_counts["regression_ee"], 1):.4f}')
        print(f"Training Loss (regression): {train_losses['regression'] / max(train_counts['regression'], 1):.4f}")
        print(f"Training Accuracy (binary): {metrics['train']['accuracies']['binary'][-1]:.4f}")
        print(f"Training Accuracy (binary_ee): {metrics['train']['accuracies']['binary_ee'][-1]:.4f}")
        print(f"Training MSE: {metrics['train']['mse']['regression'][-1]:.4f}")
        print(f"Training MSE (ee): {metrics['train']['mse']['ee'][-1]:.4f}")


        # Fase de validación
        model.eval()
        val_losses = {"binary_ee": 0.0, "regression_ee": 0.0, "binary": 0.0, "regression": 0.0}
        val_counts = {key: 0 for key in val_losses.keys()}
        correct_binary_ee_val, correct_binary_val = 0, 0
        mse_val_ee, mse_val = 0.0, 0.0
        total_loss_val = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images = images.to(device)

                # Forward pass en el conjunto de validación
                binary_pred_ee, regression_pred_ee, binary_pred, regression_pred = model(images)
                regression_pred = regression_pred.squeeze(-1)
                regression_pred_ee = regression_pred_ee.squeeze(-1)

                # Preparar etiquetas y máscaras
                binary_labels = torch.tensor([label['binary'] for label in labels]).to(device)
                regression_labels = torch.tensor([label['regression'] for label in labels]).to(device)

                binary_mask = binary_labels != -9999
                regression_mask = regression_labels != -9999.0

                # Calcular pérdidas en validación
                total_loss_batch = torch.tensor(0.0, device=device)

                if binary_mask.any():
                    loss_binary = criterion_binary(
                        binary_pred[binary_mask].float(),
                        binary_labels[binary_mask].float().unsqueeze(-1)
                    )
                    val_losses["binary"] += loss_binary.item()
                    val_counts["binary"] += binary_mask.sum().item()
                    total_loss_batch += loss_binary

                    probabilities = torch.sigmoid(binary_pred[binary_mask].float())
                    binary_predictions = (probabilities > 0.5).float()
                    correct_binary_val += (binary_predictions == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()

                    loss_binary_ee = criterion_binary(
                        binary_pred_ee[binary_mask].float(),
                        binary_labels[binary_mask].float().unsqueeze(-1)
                    )
                    val_losses["binary_ee"] += loss_binary_ee.item()
                    val_counts["binary_ee"] += binary_mask.sum().item()
                    total_loss_batch += loss_binary_ee

                    probabilities_ee = torch.sigmoid(binary_pred_ee[binary_mask].float())
                    binary_predictions_ee = (probabilities_ee > 0.5).float()
                    correct_binary_ee_val += (binary_predictions_ee == binary_labels[binary_mask].float().unsqueeze(-1)).sum().item()

                if regression_mask.any():
                    loss_regression = criterion_regression(
                        regression_pred[regression_mask].float(),
                        regression_labels[regression_mask].float()
                    )
                    val_losses["regression"] += loss_regression.item()
                    val_counts["regression"] += regression_mask.sum().item()
                    total_loss_batch += loss_regression

                    mse_val += loss_regression.item() * regression_mask.sum().item()

                    loss_regression_ee = criterion_regression(
                        regression_pred_ee[regression_mask].float(),
                        regression_labels[regression_mask].float()
                    )
                    val_losses["regression_ee"] += loss_regression_ee.item()
                    val_counts["regression_ee"] += regression_mask.sum().item()
                    total_loss_batch += loss_regression_ee

                    mse_val_ee += loss_regression_ee.item() * regression_mask.sum().item()

                total_loss_val += total_loss_batch.item()

        # Guardar métricas de validación
        avg_loss_val = total_loss_val / max(sum(val_counts.values()), 1)
        metrics["val"]["losses"].append(avg_loss_val)
        metrics["val"]["accuracies"]["binary_ee"].append(correct_binary_ee_val / max(val_counts["binary_ee"], 1))
        metrics["val"]["accuracies"]["binary"].append(correct_binary_val / max(val_counts["binary"], 1))
        metrics["val"]["mse"]['regression'].append(mse_val / max(val_counts["regression"], 1))
        metrics["val"]["mse"]['ee'].append(mse_val_ee / max(val_counts["regression_ee"], 1))

        # Imprimir métricas de validación
        print(f"Validation Loss: {avg_loss_val:.4f}")
        print(f"Validation Loss (binary_ee): {val_losses['binary_ee'] / max(val_counts['binary_ee'], 1):.4f}")
        print(f"Validation Loss (binary): {val_losses['binary'] / max(val_counts['binary'], 1):.4f}")
        print(f"Validation Loss (regression_ee): {val_losses['regression_ee'] / max(val_counts['regression_ee'], 1):.4f}")
        print(f"Validation Loss (regression): {val_losses['regression'] / max(val_counts['regression'], 1):.4f}")
        print(f"Validation Accuracy (binary_ee): {metrics['val']['accuracies']['binary_ee'][-1]:.4f}")
        print(f"Validation Accuracy (binary): {metrics['val']['accuracies']['binary'][-1]:.4f}")
        print(f"Validation MSE: {metrics['val']['mse']['regression'][-1]:.4f}")
        print(f"Validation MSE (ee): {metrics['val']['mse']['ee'][-1]:.4f}")


    # Guardar métricas en un archivo pickle
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
        # Convierte cada imagen en un tensor de PyTorch (si no se hizo antes)
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)  # Canal, alto, ancho
        label = self.labels[idx]
        return image, label

# Función collate personalizada para manejar None en etiquetas
def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), labels





def dataset_creator(path_to_folder, max_by_experiment=3):
    col, steer = 0, 0, 
    images, labels = [], []

    for folder in os.listdir(path_to_folder):
        folder_path = os.path.join(path_to_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                labels_txt_path = os.path.join(folder_path, file)

                # Procesar archivos 'labels' (binary classification)
                if file.endswith('.txt') and 'labels' in file and col < max_by_experiment:
                    col += 1
                    img_list = []
                    img_dir = os.path.join(folder_path, 'images')
                    
                    # Cargar imágenes y normalizarlas
                    for pic in sorted(os.listdir(img_dir)):
                        img = load_img(os.path.join(img_dir, pic), target_size=(200, 200), color_mode='grayscale')
                        img_array = (img_to_array(img) / 128.0) - 1.0  # Normalizar la imagen
                        img_list.append(img_array)
                    
                    # Cargar etiquetas y añadir datos
                    labels_txt = np.loadtxt(labels_txt_path)
                    for img_array, label_val in zip(img_list, labels_txt):
                        images.append(img_array)
                        label = {
                            "obj_detect": torch.zeros(4),  # Placeholder para detección de objetos
                            "binary": torch.tensor(label_val) if label_val in [0, 1] else -9999.0,  # Clasificación binaria
                            "regression": -9999.0  # Usar -9999.0 para la regresión
                        }
                        labels.append(label)



                # Procesar archivos 'sync' (regression)
                elif file.endswith('.txt') and 'sync' in file and steer < max_by_experiment:
                    steer += 1
                    img_list = []
                    img_dir = os.path.join(folder_path, 'images')
                    for pic in sorted(os.listdir(img_dir)):
                        img = load_img(os.path.join(img_dir, pic), target_size=(200, 200), color_mode='grayscale')
                        img_array = (img_to_array(img) / 128.0) - 1.0  # Normalizar la imagen
                        img_list.append(img_array)

                    labels_txt = np.loadtxt(labels_txt_path, usecols=0, delimiter=',', skiprows=1)
                    for img_array, label_val in zip(img_list, labels_txt):
                        images.append(img_array)
                        label = {
                            "obj_detect": torch.zeros(4),  # Placeholder para detección de objetos
                            "binary": -9999.0,  # Placeholder para clasificación binaria
                            "regression": torch.tensor(label_val, dtype=torch.float32) if not np.isnan(label_val) else -9999.0  # Regresión
                        }
                        labels.append(label)

    return images, labels


images_train, labels_train = dataset_creator('../data/merged_data/training', 100)
images_val, labels_val = dataset_creator('../data/merged_data/validation', 20)

indices = np.random.permutation(len(images_train))
images_train = [images_train[i] for i in indices]
labels_train = [labels_train[i] for i in indices]

# Mezclar datos de validación
indices = np.random.permutation(len(images_val))
images_val = [images_val[i] for i in indices]
labels_val = [labels_val[i] for i in indices]



train_dataset = MultiTaskDataset(images=images_train, labels=labels_train)
val_dataset = MultiTaskDataset(images=images_val, labels=labels_val)

# Crear dataloaders de entrenamiento y validación
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


trained_model, _ = train_model(train_dataloader, val_dataloader, model,optimizer, criterion_regression, criterion_binary, epochs=10)


trained_model.to('cpu')
torch.save(trained_model.state_dict(), 'trained_model.pth')

dummy_input = torch.randn(1, 1, 200, 200)
torch.onnx.export(trained_model, dummy_input, "trained_model.onnx", input_names=['input'], output_names=['ee_branch_binary', 'ee_branch_reg', 'binary_output', 'regression_output'])


class CalibrationDataReader:
    def __init__(self, calibration_dir):
        self.calibration_files = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.endswith('.npy')]
        self.data_index = 0
    
    def get_next(self):
        if self.data_index < len(self.calibration_files):
            input_data = np.load(self.calibration_files[self.data_index])
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
            self.data_index += 1
            return {'input': input_data.astype(np.float32)}
        else:
            return None

    def rewind(self):
        self.data_index = 0

# Path to the calibration data
calibration_dir = './calibration_data'

# Set up calibration data reader
calibration_data_reader = CalibrationDataReader(calibration_dir)


onnx_model_path = 'trained_model.onnx'
onnx_quantized_model_path = 'quantized_model.onnx'

quantize_static(
    onnx_model_path,                      # Input ONNX model
    onnx_quantized_model_path,            # Output ONNX quantized model
    calibration_data_reader,              # Calibration data reader
    quant_format=QuantType.QInt8,         # Quantization format (QInt8 for full int8 quantization)
    weight_type=QuantType.QInt8           # Quantize weights and activations
)

print(f"Quantized model saved to {onnx_quantized_model_path}")


print(f'Script finalizado en fecha y hora: {datetime.now()}')


