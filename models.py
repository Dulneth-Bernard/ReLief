import torch
import torch.nn as nn
import torchvision

#Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 224, 224)):
        super(SimpleBNConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Dynamically infer the input size for the first linear layer
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            self._flattened_size = out.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class FiveCropModelWrapper(nn.Module):
    """
    Wraps a base model to handle 5-crop input. Averages predictions over the 5 crops.
    Input shape: [batch, 5, 3, 200, 300]
    otherwise gives error xpected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [64, 5, 3, 200, 300
    Output: [batch, num_classes]
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # x: [batch, 5, 3, 200, 300]
        batch_size, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)  # [batch*5, 3, 200, 300]
        logits = self.base_model(x)  # [batch*5, num_classes]
        logits = logits.view(batch_size, ncrops, -1)  # [batch, 5, num_classes]
        logits = logits.mean(dim=1)  # average over crops
        return logits

def get_pretrained_resnet18(num_classes, freeze_features=True):
    """
    Returns a ResNet18 model from torchvision, adapted for num_classes.
    If freeze_features=True, all layers except the final fully connected layer are frozen.
    """
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# Town models
class DeeperCNN(nn.Module):
    #Custom CNN, Adam, lr=0.001, with augmentation
    def __init__(self, num_classes, input_shape=(3, 224, 224)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            self._flattened_size = out.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x