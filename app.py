import pickle
import streamlit as st
import pandas as pd

# Charger le modèle sauvegardé avec le nouveau cache
@st.cache_data
def load_model():
    # Charger le modèle depuis le fichier .pkl
    return pickle.load(open("car_price_gbr.pkl", "rb"))

# Charger le modèle
model_regression = load_model()

# Afficher une confirmation
st.write("Modèle chargé avec succès!")

# Titre de l'application
st.title("Prédiction du prix d'une voiture")

# Ajouter des champs pour les entrées utilisateur
seller = st.selectbox("Vendeur", ["private", "dealer"])
offer_type = st.selectbox("Type d'offre", ["offer", "request"])
abtest = st.selectbox("AB Test", ["control", "treatment"])
vehicle_type = st.selectbox("Type de véhicule", ["sedan", "coupe", "hatchback", "convertible", "wagon", "bus", "van"])
year_of_registration = st.number_input("Année d'enregistrement", min_value=1900, max_value=2024, value=2015)
gearbox = st.selectbox("Boîte de vitesses", ["manual", "automatic"])
power_ps = st.number_input("Puissance en PS", min_value=1, max_value=1000, value=120)
model = st.text_input("Modèle", "Golf")
kilometer = st.number_input("Kilométrage", min_value=0, max_value=500000, value=150000)
fuel_type = st.selectbox("Type de carburant", ["petrol", "diesel", "lpg", "cng", "hybrid", "electric"])
brand = st.text_input("Marque", "Volkswagen")
not_repaired_damage = st.selectbox("Dommages non réparés", ["no", "yes"])
age = st.number_input("Âge de la voiture", min_value=0, max_value=100, value=8)

# Créer un DataFrame avec les données de l'utilisateur
user_data = pd.DataFrame({
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

# Faire la prédiction
if st.button("Prédire le prix"):
    prediction = model_regression.predict(user_data)
    st.write(f"Prix prédit : {prediction[0]:.2f} EUR")
