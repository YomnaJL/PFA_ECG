import torch
import torch.nn as nn
import torchvision.models as models
import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image

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



def single_prediction(model, image_path, tabular_data, scaler, device):
    """
    Fonction pour effectuer une prédiction sur une image avec des données tabulaires associées.

    Arguments:
        model (nn.Module): Modèle entraîné.
        image_path (str): Chemin de l'image à prédire.
        tabular_data (list): Données tabulaires associées à l'image (ex: âge, sexe, etc.).
        scaler (scaler): Le scaler utilisé pour normaliser les données tabulaires.
        device (torch.device): Le périphérique pour exécuter les calculs (CPU ou GPU).

    Retourne:
        pd.DataFrame: DataFrame avec les prédictions du modèle pour chaque caractéristique.
    """
    # Charger et transformer l'image
    transform_pred = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner à la taille attendue par le modèle
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalisation
    ])

    image = Image.open(image_path).convert("RGB")  # Charger l'image et convertir en RGB
    image_tensor = transform_pred(image).unsqueeze(0).to(device)  # Ajouter une dimension de batch et envoyer sur le bon device

    # Appliquer le scaler aux données tabulaires (normalisation)
    tabular_data = np.array(tabular_data).reshape(1, -1)  # Reshaper les données pour correspondre à la forme attendue par le scaler
    tabular_data = scaler.transform(tabular_data)  # Appliquer la transformation (normalisation)
    tabular_data = torch.tensor(tabular_data, dtype=torch.float32).to(device)  # Convertir en tenseur et envoyer sur le bon device

    # Mettre le modèle en mode évaluation
    model.eval()
    
    with torch.no_grad():
        # Faire la prédiction en utilisant l'image et les données tabulaires
        prediction = model(image_tensor, tabular_data).cpu().numpy().flatten()

    # Labels correspondant aux différentes sorties
    labels = [
        "P_Amp_V1", "P_Amp_II", "P_Amp_I", "P_Dur_Global", "PR_Int_Global",
        "QT_IntFramingham_Global", "S_Amp_Global", "PQ_Int_Global",
        "QRS_Dur_Global", "RR_Mean_Global", "QT_Int_Global"
    ]

    # Créer un DataFrame avec les résultats
    df_results = pd.DataFrame({"Label": labels, "Valeur prédite": prediction})

    return df_results




# Define the LightCNN model 
class LightCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, target_columns=None):
        super().__init__()
        self.target_columns = target_columns or [
            "Myocardial Infarction (MI)", "ST/T Change (STTC)",
            "Conduction Disturbance (CD)", "Hypertrophy (HYP)", "Normal (NORM)"
        ]

        def dw_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, out_c, kernel_size=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        self.conv1 = dw_conv(3, 16)
        self.conv2 = dw_conv(16, 32)
        self.conv3 = dw_conv(32, 64)

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.mlp = nn.Sequential(
            nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32), nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, len(self.target_columns))
        )

    def forward(self, img, tab):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cnn_fc(x)
        y = self.mlp(tab)
        return self.classifier(torch.cat([x, y], dim=1))



def classify_image_and_tabular_data(image, tab_data, model, device, threshold=0.34):
    """
    Fonction pour classer une image avec des données tabulaires associées en utilisant le modèle LightCNN.
    
    Args:
    - image: Image d'entrée.
    - tab_data: Données tabulaires associées à l'image (ex: âge, sexe, etc.).
    - model: Le modèle LightCNN pré-entrainé.
    - device: Le périphérique (CPU ou GPU) pour exécuter le modèle.
    - threshold: Le seuil pour convertir les probabilités en valeurs binaires (par défaut 0.3).
    
    Returns:
    - dict: Dictionnaire des classes avec les résultats binaires (0 ou 1) après seuil.
    """
    target_columns = model.target_columns

    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Transformez l'image et déplacez-la sur le bon périphérique
    image = image_transform(image).unsqueeze(0).to(device)  # (1, 3, 128, 128)

    # Assurez-vous que les données tabulaires sont sur le même périphérique que le modèle
    tab_data = tab_data.to(device)

    # Exécution du modèle en mode évaluation
    model.eval()
    with torch.no_grad():
        output = model(image, tab_data)
        probs = torch.sigmoid(output).cpu().numpy()[0]  # Obtenez les probabilités
    
    # Appliquer un seuil de 0.34 pour obtenir des valeurs binaires (0 ou 1)
    binary_results = {column: 1 if prob > threshold else 0 for column, prob in zip(target_columns, probs)}

    return binary_results  # Retourner les résultats binaires
