import streamlit as st
import pickle
import pandas as pd

import os

# Charger le modèle en utilisant le chemin absolu
model_regression = pickle.load(open(os.path.join(os.getcwd(), "car_price_gbr.pkl"), mode="rb"))

# Interface Streamlit
st.title("Prédiction du prix d'une voiture")
seller = st.selectbox("Type de vendeur", ['private', 'dealer'])
offer_type = st.selectbox("Type d'offre", ['offer', 'wanted'])
abtest = st.selectbox("A/B Test", ['control', 'treatment'])
vehicle_type = st.selectbox("Type de véhicule", ['sedan', 'suv', 'coupe', 'convertible', 'hatchback'])
year_of_registration = st.number_input("Année d'enregistrement", min_value=1900, max_value=2024)
gearbox = st.selectbox("Boîte de vitesse", ['manual', 'automatic'])
power_ps = st.number_input("Puissance (PS)", min_value=0, max_value=1000)
model = st.text_input("Modèle")
kilometer = st.number_input("Kilomètres", min_value=0, max_value=500000)
fuel_type = st.selectbox("Type de carburant", ['petrol', 'diesel', 'electric', 'lpg'])
brand = st.text_input("Marque")
not_repaired_damage = st.selectbox("Dommages non réparés", ['no', 'yes'])
age = st.number_input("Âge du véhicule", min_value=0, max_value=50)

# Créer un dataframe pour la prédiction
data_unseen = pd.DataFrame({
    'index': [1],
    'seller': [seller],
    'offerType': [offer_type],
    'abtest': [abtest],
    'vehicleType': [vehicle_type],
    'yearOfRegistration': [year_of_registration],
    'gearbox': [gearbox],
    'powerPS': [power_ps],
    'model': [model],
    'kilometer': [kilometer],
    'fuelType': [fuel_type],
    'brand': [brand],
    'notRepairedDamage': [not_repaired_damage],
    'age': [age]
})

# Faire une prédiction
prediction = model_regression.predict(data_unseen)
st.write(f"Prix prédit : {prediction[0]:.2f} €")
