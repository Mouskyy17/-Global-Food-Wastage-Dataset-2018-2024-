import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Charger les objets sauvegardés
scaler = joblib.load('scaler.joblib')
lasso_model = joblib.load('lasso_model.joblib')
label_encoders = {
    'Country': joblib.load('label_encoder_Country.joblib'),
    'Food Category': joblib.load('label_encoder_Food_Category.joblib')
}

# Interface utilisateur avec sidebar
st.title("Prédiction des pertes économiques alimentaires")
st.sidebar.header("Entrez les valeurs des variables")

# Inputs utilisateur
country = st.sidebar.selectbox("Pays", label_encoders['Country'].classes_)
year = st.sidebar.number_input("Année", min_value=2000, max_value=2030, value=2024)
food_category = st.sidebar.selectbox("Catégorie alimentaire", label_encoders['Food Category'].classes_)
total_waste = st.sidebar.number_input("Gaspillage total (tonnes)", min_value=0.0, value=1000.0)
avg_waste_per_capita = st.sidebar.number_input("Gaspillage moyen par habitant (kg)", min_value=0.0, value=50.0)
population = st.sidebar.number_input("Population (millions)", min_value=0.1, value=10.0)
household_waste = st.sidebar.number_input("Gaspillage des ménages (%)", min_value=0.0, max_value=100.0, value=30.0)

# Encodage des variables catégorielles
encoded_country = label_encoders['Country'].transform([country])[0]
encoded_food_category = label_encoders['Food Category'].transform([food_category])[0]

# Préparation des données pour la prédiction
input_data = np.array([[encoded_country, year, encoded_food_category, total_waste, avg_waste_per_capita, population, household_waste]])

# Vérification de la forme des données
st.write(f"Input shape: {input_data.shape}, Expected: {scaler.n_features_in_}")

# Assurer la cohérence du nombre de features
if input_data.shape[1] == scaler.n_features_in_:
    input_data_scaled = scaler.transform(input_data)
    # Prédiction
    if st.sidebar.button("Prédire"):
        prediction = lasso_model.predict(input_data_scaled)[0]
        st.success(f"Pertes économiques estimées : {prediction:,.2f} millions de dollars")
else:
    st.error("Le nombre de caractéristiques d'entrée ne correspond pas à celui attendu par le modèle.")
