"""
Module pour définir et charger le modèle DinoClassifier.

Ce fichier contient la définition du modèle `DinoClassifier`, basé sur un
backbone DINOv2 et une tête linéaire pour la classification. Il inclut
également une fonction `load_model` pour charger un modèle pré-entraîné à
partir d'un fichier .pth.
"""
import io

import s3fs
import torch
import torch.nn as nn


class DinoClassifier(nn.Module):
    """
    Classificateur basé sur un backbone DINOv2 et une tête linéaire.

    Args:
        backbone (nn.Module): Modèle DINOv2 pré-entraîné utilisé comme
        extracteur de caractéristiques.
        num_classes (int): Nombre de classes de sortie.

    Attributes:
        backbone (nn.Module): Le backbone DINOv2.
        head (nn.Linear): Couche de classification linéaire.
    """

    def __init__(self, backbone, num_classes):
        """Initialise le modèle avec un backbone et une couche de sortie linéaire."""
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        """
        Propagation avant du modèle.

        Args:
            x (torch.Tensor): Image d'entrée sous forme de tenseur.

        Returns:
            torch.Tensor: Logits de sortie pour chaque classe.
        """
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x

    def load_model(model_path="dinov2_classifier.pth"):
        """
        Charge un modèle depuis un fichier local ou S3.

        Args:
            model_path (str): Chemin vers le modèle (chemin local ou s3://...).

        Returns:
            DinoClassifier: Modèle chargé.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = DinoClassifier(dino_backbone, num_classes=2).to(device)

        if model_path.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(model_path, "rb") as f:
                buffer = io.BytesIO(f.read())
                model.load_state_dict(torch.load(buffer, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        model.eval()
        return model
