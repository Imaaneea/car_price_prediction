import os
import pickle
import pandas as pd
import requests
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# URL pour télécharger le fichier modèle depuis GitHub
file_url = "https://github.com/Imaaneea/your-repo/raw/refs/heads/master/car_price_rf.pkl"  # Remplacez par l'URL correcte

# Titre de l'application
st.write("""
# MSDE5 : ML Project
## Car Price Prediction App

This app predicts the **Car Price** using a machine learning model.
""")

# Afficher une image du projet dans la barre latérale
st.sidebar.image("https://miro.medium.com/v2/resize:fit:1370/1*yjQ2safdygCd9y4HbA48oA.png", width=300)
st.sidebar.header('User Input Parameters')

# Définir les paramètres d'entrée pour la prédiction
def user_input_features():
    vehicleType = st.sidebar.selectbox('Vehicle Type', ['car', 'suv', 'van', 'coupe', 'wagon', 'convertible', 'bus'])
    yearOfRegistration = st.sidebar.slider('Year of Registration', 2000, 2024, 2015)
    gearbox = st.sidebar.selectbox('Gearbox', ['manual', 'automatic'])
    powerPS = st.sidebar.slider('Power (in PS)', 50, 500, 100)
    model = st.sidebar.text_input('Model', 'Focus')
    kilometer = st.sidebar.slider('Mileage (in km)', 0, 300000, 50000)
    fuelType = st.sidebar.selectbox('Fuel Type', ['petrol', 'diesel', 'lpg', 'cng', 'electric', 'hybrid'])
    brand = st.sidebar.selectbox('Brand', ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'BMW', 'Audi'])
    notRepairedDamage = st.sidebar.selectbox('Not Repaired Damage', ['yes', 'no'])
    age = st.sidebar.slider('Age of the Vehicle', 0, 30, 5)
    
    data = {
        'vehicleType': vehicleType,
        'yearOfRegistration': yearOfRegistration,
        'gearbox': gearbox,
        'powerPS': powerPS,
        'model': model,
        'kilometer': kilometer,
        'fuelType': fuelType,
        'brand': brand,
        'notRepairedDamage': notRepairedDamage,
        'age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Obtenir les entrées utilisateur
df = user_input_features()

# Afficher les paramètres d'entrée
st.subheader('User Input Parameters')
st.write(df)

# Chemin local du fichier modèle
file_path = os.path.join(os.getcwd(), "car_price_rf.pkl")

# Charger ou télécharger le modèle
try:
    # Si le modèle n'existe pas localement, le télécharger
    if not os.path.exists(file_path):
        with st.spinner("Téléchargement du modèle, veuillez patienter..."):
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(file_path, mode="wb") as model_file:
                    model_file.write(response.content)
            else:
                st.error(f"Échec du téléchargement. Code de réponse : {response.status_code}")
                st.stop()

    # Charger le modèle depuis le fichier local
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    st.success("Le modèle a été chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement ou du téléchargement du modèle : {e}")
    st.stop()

# Prédire le prix de la voiture lorsque l'utilisateur appuie sur le bouton
if st.sidebar.button('Predict Price'):
    try:
        prediction = model.predict(df)
        st.write(f'Predicted Price: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
