import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Charger les objets sauvegardés
scaler = joblib.load('scaler.joblib')
lasso_model = joblib.load('lasso_model.joblib')
logistic_model = joblib.load('logistic_regression_model.joblib')
label_encoders = {
    'Country': joblib.load('label_encoder_Country.joblib'),
    'Food Category': joblib.load('label_encoder_Food_Category.joblib')
}

# Interface utilisateur
st.title("📉 Prédiction des Pertes Économiques et Classification du Gaspillage Alimentaire")
st.markdown("""
Bienvenue dans cette application qui prédit les pertes économiques liées au gaspillage alimentaire
et classe le niveau de gaspillage en **Faible, Moyen ou Élevé**.
Veuillez entrer les données et cliquez sur **Prédire**.
""")

st.sidebar.header("📝 Entrez les valeurs des variables")

# Informations Générales
st.sidebar.subheader("📍 Informations Générales")
country = st.sidebar.selectbox("🌍 Pays", label_encoders['Country'].classes_)
year = st.sidebar.number_input("📅 Année", min_value=2000, max_value=2030, value=2024)
food_category = st.sidebar.selectbox("🍎 Catégorie alimentaire", label_encoders['Food Category'].classes_)

st.sidebar.subheader("♻️ Données de Gaspillage")
total_waste = st.sidebar.number_input("🗑️ Gaspillage total (tonnes)", min_value=0.0, value=1000.0)
avg_waste_per_capita = st.sidebar.number_input("👤 Gaspillage moyen par habitant (kg)", min_value=0.0, value=50.0)

t.sidebar.subheader("🏠 Données Démographiques")
population = st.sidebar.number_input("👥 Population (millions)", min_value=0.1, value=10.0)
household_waste = st.sidebar.number_input("🏡 Gaspillage des ménages (%)", min_value=0.0, max_value=100.0, value=30.0)

# Encodage des variables catégorielles
encoded_country = label_encoders['Country'].transform([country])[0]
encoded_food_category = label_encoders['Food Category'].transform([food_category])[0]

# Préparation des données pour la prédiction
input_data = np.array([[encoded_country, year, encoded_food_category, avg_waste_per_capita, population, household_waste]])

# Assurer la cohérence du nombre de features
if input_data.shape[1] == scaler.n_features_in_:
    input_data_scaled = scaler.transform(input_data)
    
    if st.sidebar.button("🔮 Prédire"):
        # Prédiction des pertes économiques
        prediction_economic_loss = lasso_model.predict(input_data_scaled)[0]
        st.success(f"💰 Pertes économiques estimées : **{prediction_economic_loss:,.2f} millions de dollars**")
        
        # Classification du gaspillage
        prediction_waste_category = logistic_model.predict(input_data_scaled)[0]
        st.info(f"📊 Niveau de gaspillage prédit : **{prediction_waste_category}**")
else:
    st.error("⚠️ Le nombre de caractéristiques d'entrée ne correspond pas à celui attendu par le modèle.")
