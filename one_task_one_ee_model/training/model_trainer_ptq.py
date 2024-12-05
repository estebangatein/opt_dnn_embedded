
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tensorflow.keras.utils import img_to_array, load_img
import torch.quantization as quantization
import torch.onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnxruntime as ort


def dataset_creator(path_to_folder, max_by_experiment=3):
    fire = 0
    images, labels = [], []

    for folder in os.listdir(path_to_folder):
        if os.path.isdir(os.path.join(path_to_folder, folder)):
            for file in os.listdir(os.path.join(path_to_folder, folder)):
                if file.endswith('.txt'):
                    if 'fire' in file and fire < max_by_experiment:
                        fire += 1
                        for pic in sorted(os.listdir(os.path.join(path_to_folder, folder, 'images'))):
                            img = load_img(os.path.join(path_to_folder, folder, 'images', pic), 
                                           target_size=(200, 200), color_mode='grayscale')
                            img_array = img_to_array(img) / 255.0  # Normalize
                            images.append(img_array)
                        
                        labels_txt = np.loadtxt(os.path.join(path_to_folder, folder, file), delimiter=' ')
                        labels.extend(labels_txt)

    return np.array(images), np.array(labels)

# Loading training and validation data
images_train, labels_train = dataset_creator('../data/merged_data/training', 100)
indices = np.random.permutation(images_train.shape[0])
images_train, labels_train = images_train[indices], labels_train[indices]

images_val, labels_val = dataset_creator('../data/merged_data/validation', 5)
indices = np.random.permutation(images_val.shape[0])
images_val, labels_val = images_val[indices], labels_val[indices]

images_train = np.transpose(images_train, (0, 3, 1, 2))
labels_train = labels_train.argmax(axis=1)

images_val = np.transpose(images_val, (0, 3, 1, 2))
labels_val = labels_val.argmax(axis=1)




class simple_model(nn.Module):
    def __init__(self, img_channels=1, output_dim_objects=4):
        super(simple_model, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=5, stride=2, padding=2)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)            
        self.fc_objects_early = nn.Linear(32 * 50 * 50, output_dim_objects)           

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)            
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)            

        self.fc_objects_final = nn.Linear(64 * 13 * 13, output_dim_objects)           

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        early_exit = x.view(x.size(0), -1)
        early_exit = self.fc_objects_early(early_exit)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        final_exit = self.fc_objects_final(x)
        
        return early_exit, final_exit



def train_model(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    metrics = {
        "train_loss": [],
        "train_acc_early": [],
        "train_acc_final": [],
        "val_loss": [],
        "val_acc_early": [],
        "val_acc_final": []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects_early = 0
        corrects_final = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            early_exit, final_exit = model(inputs.float())
            loss = criterion(early_exit, labels) + criterion(final_exit, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            corrects_early += (early_exit.argmax(1) == labels).sum().item()
            corrects_final += (final_exit.argmax(1) == labels).sum().item()
            total += labels.size(0)



        metrics["train_loss"].append(running_loss / total)
        metrics["train_acc_early"].append(corrects_early / total)
        metrics["train_acc_final"].append(corrects_final / total)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {running_loss / total}")
        print(f"Train Acc Early: {corrects_early / total}")
        print(f"Train Acc Final: {corrects_final / total}")


        model.eval()
        val_loss = 0.0
        val_corrects_early = 0
        val_corrects_final = 0
        val_total = 0

        with torch.no_grad():  
            for inputs, labels in val_loader:
                early_exit, final_exit = model(inputs.float())
                loss = criterion(early_exit, labels) + criterion(final_exit, labels)

                val_loss += loss.item()
                val_corrects_early += (early_exit.argmax(1) == labels).sum().item()
                val_corrects_final += (final_exit.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        # Store validation metrics
        metrics["val_loss"].append(val_loss / val_total)
        metrics["val_acc_early"].append(val_corrects_early / val_total)
        metrics["val_acc_final"].append(val_corrects_final / val_total)

        print(f"Validation Loss: {val_loss / val_total}")
        print(f"Validation Acc Early: {val_corrects_early / val_total}")
        print(f"Validation Acc Final: {val_corrects_final / val_total}")


    return model






# Prepare DataLoaders
X_train = torch.from_numpy(images_train)
y_train = torch.from_numpy(labels_train)
X_val = torch.from_numpy(images_val)
y_val = torch.from_numpy(labels_val)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Create DataLoader for calibration data
X_calibration = torch.from_numpy(images_val[:100])  # Using a subset for calibration
y_calibration = torch.from_numpy(labels_val[:100])
calibration_dataset = TensorDataset(X_calibration, y_calibration)
calibration_loader = DataLoader(calibration_dataset, batch_size=16, shuffle=False)

# Initialize model
model = simple_model()


trained_model = train_model(model, train_loader, val_loader, num_epochs=100)


dummy_input = torch.randn(1, 1, 200, 200)
onnx_model_path = 'model_trained.onnx'
torch.onnx.export(trained_model, dummy_input, onnx_model_path, input_names=['input'], output_names=['early_exit', 'final_exit'])


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


onnx_model_path = 'model_trained.onnx'
onnx_quantized_model_path = 'quantized_model.onnx'

quantize_static(
    onnx_model_path,                      # Input ONNX model
    onnx_quantized_model_path,            # Output ONNX quantized model
    calibration_data_reader,              # Calibration data reader
    quant_format=QuantType.QInt8,         # Quantization format (QInt8 for full int8 quantization)
    weight_type=QuantType.QInt8           # Quantize weights and activations
)

print(f"Quantized model saved to {onnx_quantized_model_path}")


ort_session = ort.InferenceSession(onnx_quantized_model_path)

# Prepare input (e.g., use one of your validation samples)
input_image = images_val[11:12]  # Assuming images_val is already loaded
input_image = input_image.astype(np.float32)

# Run inference on the quantized model
outputs = ort_session.run(None, {'input': input_image})

early_exit, final_exit = outputs
print("Early exit:", early_exit)
print("Final exit:", final_exit)


