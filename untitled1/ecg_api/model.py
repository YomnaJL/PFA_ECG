import torch
import torch.nn as nn
import torchvision.models as models

class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        # Utiliser ResNet18 
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)

        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 11)
        )

    def forward(self, image, tabular):
        x1 = self.resnet(image)
        x2 = self.mlp(tabular)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


