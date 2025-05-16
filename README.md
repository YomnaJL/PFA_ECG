# Système Intelligent d'Analyse et de Diagnostic des ECG

Ce projet a pour objectif de développer un système intelligent permettant l'analyse et le diagnostic des électrocardiogrammes (ECG). Une application mobile réalisée en Flutter intègre deux modèles de deep learning complémentaires :

- **Modèle de Régression** : Estime les mesures cliniques d’un ECG (amplitude et durée des pics R, T, intervalles PR, QT, etc.) à partir d’une image ECG et de données patient.
- **Modèle de classification multi-label** : Classifie les ECG en 5 catégories de maladies cardiaques.

## 📱 Description du Projet

Ce projet a pour objectif d’exploiter le deep learning pour l’analyse des ECG via une application mobile. Nous avons :

- 💡 Développé une application **Flutter** intégrant deux modèles de deep learning.
- 📈 Conçu un **modèle de prédiction** pour estimer les mesures cliniques d’un ECG.
- 🏥 Conçu un **modèle de classification multi-label** pour détecter 5 pathologies cardiaques.
- 🧼 Mis en place un **pipeline de prétraitement** pour les images ECG et les données tabulaires.
- ✅ Obtenu des **performances élevées** avec une précision de `0.91` pour la classification et des métriques prometteuses pour la prédiction.

---


## Installation et Dépendances

Pour exécuter le projet, installez les dépendances suivantes :

```bash
pip install flask==2.2.2
pip install torch==1.12.0
pip install torchvision==0.13.0
pip install pillow==9.2.0
pip install joblib==1.1.0
pip install pandas==1.4.2
pip install numpy==1.23.2 
```
# 📊 Performance des Modèles

## 1. 🔢 Modèle de Prédiction

### Comparaison des Métriques selon le Type d'Image ECG

| Type d’image                  | RMSE    | MAE     | R²     | Inversion de signe (%) |
|------------------------------|---------|---------|--------|-------------------------|
| Image originale              | 19.9658 | 10.9190 | 0.9105 | 15.69                   |
| Fond noir / signal blanc     | 19.9461 | 11.0439 | 0.9021 | 13.20                   |
| Fond blanc / signal noir     | 20.3309 | 11.2324 | 0.9113 | 13.38                   |

---

## 2. 🧠 Modèle de Classification

### Rapport de Classification

| Classe                          | Précision | Rappel | F1-score | Support |
|--------------------------------|-----------|--------|----------|---------|
| Myocardial Infarction (MI)     | 0.79      | 0.35   | 0.49     | 214     |
| ST/T Change (STTC)             | 0.73      | 0.70   | 0.72     | 247     |
| Conduction Disturbance (CD)    | 0.84      | 0.75   | 0.79     | 83      |
| Hypertrophy (HYP)              | 0.88      | 0.50   | 0.64     | 149     |
| Normal (NORM)                  | 0.91      | 0.91   | 0.91     | 565     |
| **Micro avg**                  | 0.85      | 0.71   | 0.78     | 1258    |
| **Macro avg**                  | 0.83      | 0.64   | 0.71     | 1258    |
| **Weighted avg**               | 0.85      | 0.71   | 0.76     | 1258    |
| **Samples avg**                | 0.67      | 0.64   | 0.65     | 1258    |

**Test Loss**: `0.2092`  
**Test Accuracy**: `0.9169`

---


## 🧪 Exemples de Régression

| Modèle                        | Exemple 1                            | Exemple 2                            |
|------------------------------|--------------------------------------|--------------------------------------|
| **📏 Modèle de Prédiction (Mesures)**     | ![Mesure1](pred-1.png)    | ![Mesure2](pred-2.png)    |
| **🩺 Modèle de Classification (Multi-label)**  | ![Classif1](exemple-classif-1.png)  | ![Classif2](exemple-classif-4.png)  |

## ⚙️ Utilisation de l'API Flask pour les Prédictions

L'API Flask vous permet d'effectuer des prédictions en temps réel en utilisant les modèles de deep learning. Voici un exemple d'utilisation de l'API Flask pour envoyer une image et obtenir la prédiction :

1. **Lancer l'API Flask** :
```bash
   python app.py
```

2. **tester l'application** :


| <img src="https://drive.google.com/uc?id=18SOjwZYRlt1Vmet3zX2Q5F4ubqKgbrLN" width="200"> | <img src="https://drive.google.com/uc?id=1R-DZjALsB7JF-IeGV-AmzyZ-ViOFoMgF" width="200"> |


🎥 [Voir la démo vidéo](cap5.mp4)

