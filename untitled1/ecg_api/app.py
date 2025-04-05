from flask import Flask, request, jsonify
import torch
import joblib
from PIL import Image
import io
import numpy as np
from model import ECGModel, single_prediction  # Assure-toi que ECGModel est bien importé

app = Flask(__name__)

# Charger le modèle et le scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGModel()
model.load_state_dict(torch.load("ecg_model.pth", map_location=device))
model.to(device)
model.eval()

scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Récupérer l'image
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    
    # Récupérer les données tabulaires
    data = request.form.get("tabular_data")  # Ex: "56.0,1,0.0,63.0"
    tabular_data = [float(x) for x in data.split(",")]

    # Faire la prédiction
    df = single_prediction(model, image, tabular_data, scaler, device)

    # Retourner le résultat au format JSON
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
