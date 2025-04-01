import torch
import torch.nn as nn
from torch.hub import load

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes) # check out_feature cua dinov2

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x

def load_model(model_path="dinov2_classifier.pth"):
    # set device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model = DinoClassifier(dino_backbone, num_classes=2).to(device)
    # model.head = nn.Linear(model.head.in_features, 2)  # Adjusting for 2 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model