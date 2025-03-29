import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition for the specified little YOLO model
class LittleYOLO(nn.Module):

    def __init__(self):

        super(LittleYOLO, self).__init__()

        # Convolution 1: 3 -> 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # 3 input channels (RGB), 16 output channels
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 2: 16 -> 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 3: 32 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 4: 64 -> 64
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution 5: 64 -> 32
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Flatten layer: 7x7 output
        self.flatten_size = 32 * 7 * 7

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # FC layer
        self.fc1 = nn.Linear(self.flatten_size, 512)

        # Output layer: 7x7 grid with 7 values per cell (2 class scores + 5 bbox values)
        self.fc_out = nn.Linear(512, 343)

        # Kaiming Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.leaky_relu(self.bn3(self.conv3(x))) 
        x = self.pool3(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = F.leaky_relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1) # Flatten
        x = self.dropout(x) # Dropout
        x = F.leaky_relu(self.fc1(x)) # FC
        x = torch.sigmoid(self.fc_out(x))  # Output with sigmoid activation

        return x

# Model definition for the modified resnet18 backbone model
class LittleYOLO_ResNet18(nn.Module):

    def __init__(self):

        super(LittleYOLO_ResNet18, self).__init__()

        # Load the pretrained ResNet-18 model
        pretrained_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Set the resnet backbone with the last FC layer stripped
        self.resnet_backbone = nn.Sequential(*list(pretrained_resnet.children())[:-1])

        # Flatten layer: resnet 112x112 input -> output 512 * 4 * 4
        self.flatten_size = 512 * 4 * 4

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # FC layer
        self.fc1 = nn.Linear(self.flatten_size, 512)

        # Output layer: 7x7 grid with 7 values per cell (2 class scores + 5 bbox values)
        self.fc_out = nn.Linear(512, 343)

        # Kaiming Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.resnet_backbone(x) # Resnet backbone

        x = torch.flatten(x, 1) # Flatten
        x = self.dropout(x) # Dropout
        x = F.leaky_relu(self.fc1(x)) # FC
        x = torch.sigmoid(self.fc_out(x))  # Output with sigmoid activation

        return x