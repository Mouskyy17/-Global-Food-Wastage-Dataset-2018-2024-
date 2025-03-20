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
food_category = st.sidebar.selectbox("Catégorie alimentaire", label_encoders['Food Category'].classes_)
food_waste = st.sidebar.number_input("Quantité de gaspillage alimentaire (tonnes)", min_value=0.0, value=1000.0)
co2_emission = st.sidebar.number_input("Émissions de CO2 (tonnes)", min_value=0.0, value=500.0)
water_use = st.sidebar.number_input("Utilisation de l'eau (m³)", min_value=0.0, value=10000.0)
land_use = st.sidebar.number_input("Utilisation des terres (hectares)", min_value=0.0, value=50.0)

# Encodage des variables catégorielles
encoded_country = label_encoders['Country'].transform([country])[0]
encoded_food_category = label_encoders['Food Category'].transform([food_category])[0]

# Préparation des données pour la prédiction
input_data = np.array([[encoded_country, encoded_food_category, food_waste, co2_emission, water_use, land_use]])
input_data_scaled = scaler.transform(input_data)

# Prédiction
if st.sidebar.button("Prédire"):
    prediction = lasso_model.predict(input_data_scaled)[0]
    st.success(f"Pertes économiques estimées : {prediction:,.2f} millions de dollars")
