"""
Application Streamlit pour la classification d'images avec DINOv2.

Cette application permet de téléverser une image et d'obtenir une prédiction
via une API FastAPI.
"""
import requests
from PIL import Image

import streamlit as st

st.set_page_config(page_title="Prédiction DINOv2", layout="centered")
st.title("Classificateur Dandelion vs Grass (DINOv2)")

st.markdown("Téléversez une image pour obtenir une prédiction via l'API FastAPI.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affiche l’image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    # Bouton pour lancer la prédiction
    if st.button("Envoyer à l'API pour prédiction"):
        try:
            with st.spinner("Envoi de l’image à l’API..."):
                response = requests.post(
                    url="http://fastapi-api:8000/predict",
                    files={"file": uploaded_file.getvalue()},
                )
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Résultat : **{prediction}**")
            else:
                st.error(f"Erreur de l'API : {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de se connecter à l'API.\
                     Vérifiez qu'elle est bien lancée sur le port 8000."
            )
