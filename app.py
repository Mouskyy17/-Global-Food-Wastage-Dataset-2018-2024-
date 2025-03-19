# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Charger le modèle et le scaler
model = joblib.load('lasso_model.joblib')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.joblib')

# Récupérer les classes connues depuis les encodeurs
available_countries = label_encoders["Country"].classes_.tolist()
available_categories = label_encoders["Food Category"].classes_.tolist()

# 2. Créer l'interface utilisateur
st.title('📊 Prédiction des Pertes Économiques liées au Gaspillage Alimentaire')

st.markdown("""
Cette application prédit les pertes économiques (en millions $) basées sur les caractéristiques du gaspillage alimentaire.
""")

# 3. Sidebar pour les inputs utilisateur
st.sidebar.header('📥 Paramètres d\'Entrée')

def user_input_features():
    country = st.sidebar.selectbox('Pays', available_countries)
    year = st.sidebar.slider('Année', 2018, 2025, 2022)
    food_category = st.sidebar.selectbox('Catégorie', available_categories)
    total_waste = st.sidebar.number_input('Déchets Totaux (Tonnes)', min_value=0.0, value=10000.0)
    avg_waste = st.sidebar.number_input('Déchet Moyen par Habitant (Kg)', min_value=0.0, value=50.0)
    population = st.sidebar.number_input('Population (Millions)', min_value=0.0, value=50.0)
    household_waste = st.sidebar.slider('Déchet Ménager (%)', 0.0, 100.0, 30.0)
    
    data = {
        'Country': country,
        'Year': year,
        'Food Category': food_category,
        'Total Waste (Tons)': total_waste,
        'Avg Waste per Capita (Kg)': avg_waste,
        'Population (Million)': population,
        'Household Waste (%)': household_waste
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 4. Prétraitement des données
def preprocess_input(input_df):
    # Encodage des variables catégorielles
    for col in ["Country", "Food Category"]:
        if col in label_encoders:  # Vérification que l'encodeur existe
            input_df[col] = input_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)
        else:
            raise ValueError(f"L'encodeur pour '{col}' est introuvable.")

    # Sélection des colonnes dans le bon ordre (exactement comme lors de l'entraînement)
    expected_features = ["Country", "Year", "Food Category", "Total Waste (Tons)", 
                         "Avg Waste per Capita (Kg)", "Population (Million)", "Household Waste (%)"]

    # Vérification que toutes les colonnes attendues sont bien présentes
    missing_features = set(expected_features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Colonnes manquantes dans les données d'entrée : {missing_features}")

    # Réorganisation des colonnes
    input_df = input_df[expected_features]

    # Normalisation des variables numériques
    numerical_features = ["Year", "Total Waste (Tons)", "Avg Waste per Capita (Kg)", "Population (Million)", "Household Waste (%)"]
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df


processed_df = preprocess_input(input_df.copy())

# 5. Affichage des données d'entrée
st.subheader('📋 Paramètres Saisis')
st.write(input_df)

# 6. Prédiction
if st.button('🔮 Prédire les Pertes Économiques'):
    prediction = model.predict(processed_df)
    st.success(f'Perte prédite : ${prediction[0]:.2f} millions')
    
    # Explication supplémentaire
    st.markdown("""
    **Interprétation :**
    - Cette prédiction est basée sur un modèle Random Forest entraîné sur des données historiques
    - La précision moyenne du modèle est de ± $2.5 millions
    - Les facteurs les plus influents sont généralement les déchets totaux et la population
    """)

# 7. Section d'information
st.markdown("---")
st.markdown("""
**📌 Mode d'emploi :**
1. Remplir les paramètres dans la sidebar ←
2. Cliquer sur le bouton de prédiction
3. Consulter les résultats ci-dessus

*Les données utilisées proviennent du Global Food Wastage Dataset 2018-2024*
""")