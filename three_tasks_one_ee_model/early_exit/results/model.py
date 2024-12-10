import torch
import torch.nn as nn
import torch.nn.functional as F


class OneEE(nn.Module):
    def __init__(self):
        super(OneEE, self).__init__()

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

        # Early exit: detección de objetos (4 clases) después del segundo bloque
        self.obj_detect_conv = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.obj_detect_bn = nn.BatchNorm2d(128)
        self.obj_detect_fc_ee = nn.Linear(128 * 50 * 50, 4)  # Tamaño reducido a 50x50 después del pooling

        # Capas adicionales para el resto de tareas
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Salida final de detección de objetos (4 clases) después del tercer bloque
        self.obj_detect_fc_final = nn.Linear(512 * 25 * 25, 4)

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

        # Early exit para detección de objetos
        obj_detect_ee = F.relu(self.obj_detect_bn(self.obj_detect_conv(x)))
        obj_detect_ee = torch.flatten(obj_detect_ee, 1)  # Aplanar
        obj_detect_ee = self.obj_detect_fc_ee(obj_detect_ee)

        # Continuamos con el resto de capas para las otras tareas
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = torch.flatten(x, 1)  # Aplanar

        # Salida final para detección de objetos
        obj_detect_final = self.obj_detect_fc_final(x)

        # Clasificación binaria
        binary_output = F.relu(self.binary_fc1(x))
        binary_output = self.binary_fc2(binary_output)

        # Regresión
        regression_output = F.relu(self.regression_fc1(x))
        regression_output = self.regression_fc2(regression_output)

        return obj_detect_ee, obj_detect_final, binary_output, regression_output