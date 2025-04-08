from flask import Flask, request, jsonify
import torch
import joblib
from PIL import Image
import os
import uuid
from model import ECGModel, single_prediction, LightCNN, classify_image_and_tabular_data

app = Flask(__name__)

# Charger le modèle ECG (pour la prédiction) et le scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ecg_model = ECGModel()
ecg_model.load_state_dict(torch.load("ecg_model.pth", map_location=device))
ecg_model.to(device)
ecg_model.eval()

scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Récupérer l'image
    image_file = request.files["image"]
    
    # Générer un nom de fichier temporaire unique
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("temp_images", temp_filename)

    # Créer le dossier temp_images s'il n'existe pas
    os.makedirs("temp_images", exist_ok=True)

    # Sauvegarder le fichier temporairement
    image_file.save(temp_path)

    # Récupérer les données tabulaires
    data = request.form.get("tabular_data")
    tabular_data = [float(x) for x in data.split(",")]

    # Faire la prédiction en utilisant ecg_model
    df = single_prediction(ecg_model, temp_path, tabular_data, scaler, device)

    # Supprimer l'image temporaire après la prédiction
    os.remove(temp_path)

    # Retourner le résultat
    return jsonify(df.to_dict(orient="records"))

# Charger le modèle de classification LightCNN (pour la classification)
classifier_model = LightCNN()
classifier_model.load_state_dict(torch.load("ecg_classifier_final.pth", map_location=device))
classifier_model.to(device)
classifier_model.eval()

@app.route("/classification", methods=["POST"])
def classify():
    # Récupérer l'image
    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")

    # Récupérer les données tabulaires
    data = request.form.get("tabular_data")
    try:
        tabular_data = [float(x) for x in data.split(",")]
    except Exception as e:
        return jsonify({"error": "Les données tabulaires doivent être des float séparés par des virgules"}), 400

    # Convertir les données tabulaires en tenseur
    tab_tensor = torch.tensor(tabular_data, dtype=torch.float32).unsqueeze(0).to(device)

    # Faire la classification en utilisant classifier_model
    result = classify_image_and_tabular_data(image, tab_tensor, classifier_model, device)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
