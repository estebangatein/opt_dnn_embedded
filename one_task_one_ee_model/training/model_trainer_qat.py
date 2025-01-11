
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tensorflow.keras.utils import img_to_array, load_img
import pandas as pd
import matplotlib.pyplot as plt
import torch.quantization as quantization


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_creator(path_to_folder, max_by_experiment = 3):
    fire = 0
    images, labels = [], []

    for folder in os.listdir(path_to_folder):

        if os.path.isdir(os.path.join(path_to_folder, folder)):

            for file in os.listdir(os.path.join(path_to_folder, folder)):

                if file.endswith('.txt'):



                    if 'fire' in file and fire < max_by_experiment:
                        fire += 1
                        for pic in sorted(os.listdir(os.path.join(path_to_folder, folder, 'images'))):
                            img = load_img(os.path.join(os.path.join(path_to_folder, folder, 'images'), pic), target_size=(200, 200), color_mode='grayscale')  
                            img_array = img_to_array(img) / 255.0  
                            images.append(img_array)
                        
                        labels_txt = np.loadtxt(os.path.join(path_to_folder, folder, file), delimiter=' ')

                        for label in labels_txt:
                            #label = [np.array([np.nan]*2), label, np.array([np.nan])]
                            #label = pad_sequences(label, dtype='float32', padding='post', value=np.nan)
                            labels.append(label)



    return np.array(images), np.array(labels)

images_train, labels_train = dataset_creator('../data/merged_data/training', 20)
indices = np.random.permutation(images_train.shape[0])
images_train = images_train[indices]
labels_train = labels_train[indices]

images_val, labels_val = dataset_creator('../data/merged_data/validation', 1)
indices = np.random.permutation(images_val.shape[0])
images_val = images_val[indices]
labels_val = labels_val[indices]



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

        self.fc_objects_final = nn.Linear(64 *13* 13, output_dim_objects)           
        
        self.flatten = nn.Flatten()

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        early_exit = self.flatten(x)
        early_exit = self.fc_objects_early(early_exit)
        early_exit = self.dequant(early_exit)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.flatten(x)
        final_exit = self.fc_objects_final(x)
        final_exit = self.dequant(final_exit)
        
        return early_exit, final_exit



def prepare_model_for_qat(model):
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)


model = simple_model().to(device)
prepare_model_for_qat(model)


X_train = torch.from_numpy(images_train)
y_train = torch.from_numpy(labels_train)
X_val = torch.from_numpy(images_val)
y_val = torch.from_numpy(labels_val)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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


num_epochs =100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    corrects_early = 0
    corrects_final = 0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()


        inputs, labels = inputs.float().to(device), labels.to(device)

        early_exit, final_exit = model(inputs)
        
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
    

device = torch.device('cpu')
model.cpu()

model.eval()
quantized_model = quantization.convert(model, inplace=False)
quantized_model.cpu()

dummy_input = torch.randn((1,1,200,200)).to(device)
torch.onnx.export(
        quantized_model,
        dummy_input,
        'simple_model.onnx',
    )
