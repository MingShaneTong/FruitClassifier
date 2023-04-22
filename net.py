import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandAugment(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net_batch_size = 1
net_lr = 0.001
net_criterion = nn.NLLLoss
net_optimizer = optim.SGD

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # defined models
        self.conv1 = nn.Conv2d(3, 64, 7, stride=3)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        # trained own model
        # convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # classification layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x