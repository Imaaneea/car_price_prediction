import os
import streamlit as st
import pickle
import pandas as pd

# Afficher le répertoire courant
st.write(f"Répertoire courant : {os.getcwd()}")  # Cela vous indiquera où l'application est exécutée

# Vérifier si le fichier du modèle existe
file_path = os.path.join(os.getcwd(), "car_price_rf.pkl")  # Crée le chemin absolu vers le fichier
st.write(f"Vérification du fichier à l'emplacement : {file_path}")

# Charger le pipeline sauvegardé
try:
    # Vérifier si le fichier existe avant de le charger
    if not os.path.exists(file_path):
        st.error(f"Le fichier {file_path} n'existe pas. Veuillez vérifier le chemin et l'emplacement du fichier.")
    else:
        # Charger le modèle
        with open(file_path, mode="rb") as model_file:
            pipeline = pickle.load(model_file)
        st.write("Le modèle a été chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")

# Titre de l'application
st.title("Prédiction du Prix des Voitures 🚗")

# Description
st.write("Cette application utilise un modèle Random Forest pour estimer le prix des voitures d'occasion.")

# Entrées utilisateur
st.header("Entrez les caractéristiques du véhicule :")
vehicle_type = st.selectbox("Type de véhicule", ["", "Limousine", "Cabriolet", "SUV", "Compact", "Van"])
year_of_registration = st.number_input("Année d'enregistrement", min_value=1900, max_value=2024, value=2015)
gearbox = st.selectbox("Type de boîte de vitesses", ["", "Manual", "Automatic"])
power_ps = st.number_input("Puissance (PS)", min_value=0, max_value=1000, value=100)
model = st.text_input("Modèle (ex. Golf, Polo, etc.)")
kilometer = st.number_input("Kilométrage (en km)", min_value=0, max_value=500000, value=50000)
fuel_type = st.selectbox("Type de carburant", ["", "Petrol", "Diesel", "Electric", "CNG", "LPG"])
brand = st.text_input("Marque (ex. BMW, Audi, etc.)")
not_repaired_damage = st.selectbox("Réparé ?", ["", "Yes", "No"])
age = 2024 - year_of_registration  # Calculer l'âge à partir de l'année d'enregistrement

# Préparer les données pour la prédiction
if st.button("Prédire le prix"):
    input_data = pd.DataFrame({
        "vehicleType": [vehicle_type],
        "yearOfRegistration": [year_of_registration],
        "gearbox": [gearbox],
        "powerPS": [power_ps],
        "model": [model],
        "kilometer": [kilometer],
        "fuelType": [fuel_type],
        "brand": [brand],
        "notRepairedDamage": [not_repaired_damage],
        "age": [age]
    })

    # Vérifier si tous les champs sont remplis
    if input_data.isnull().any().any() or "" in input_data.values:
        st.error("Veuillez remplir tous les champs avant de prédire.")
    else:
        try:
            # Prédire avec le modèle chargé
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Le prix estimé du véhicule est : {prediction:.2f} unités monétaires")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
