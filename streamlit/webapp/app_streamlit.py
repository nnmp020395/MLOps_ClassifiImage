"""
Application Streamlit pour la classification d'images avec DINOv2.

Cette application permet de téléverser une image et d'obtenir une prédiction
via une API FastAPI.
"""
import requests
import threading
import time
from PIL import Image
import streamlit as st
from prometheus_client import Counter, start_http_server, REGISTRY

# ------------------ METRICS SETUP ------------------

def setup_metrics():
    try:
        page_views = REGISTRY._names_to_collectors["streamlit_page_views"]
    except KeyError:
        page_views = Counter("streamlit_page_views", "Nombre de visites de la page Streamlit")

    try:
        button_clicks = REGISTRY._names_to_collectors["predict_button_clicks"]
    except KeyError:
        button_clicks = Counter("predict_button_clicks", "Nombre de clics sur le bouton de prédiction")

    return page_views, button_clicks

# PAGE_VIEWS = Counter("streamlit_page_views", "Nombre de visites de la page Streamlit")
# BUTTON_CLICKS = Counter("predict_button_clicks", "Nombre de clics sur le bouton de prédiction")

def start_metrics_server():
    start_http_server(8502)
    while True:
        time.sleep(10)  # Keeps the thread alive

# Démarrer le serveur Prometheus en arrière-plan une seule fois
if "metrics_started" not in st.session_state:
    st.session_state.PAGE_VIEWS, st.session_state.BUTTON_CLICKS = setup_metrics()
    threading.Thread(target=start_metrics_server, daemon=True).start()
    st.session_state.metrics_started = True

PAGE_VIEWS = st.session_state.PAGE_VIEWS
BUTTON_CLICKS = st.session_state.BUTTON_CLICKS

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Prédiction DINOv2", layout="centered")
st.title("Classificateur Dandelion vs Grass (DINOv2)")

PAGE_VIEWS.inc()  # Increment the page view counter

st.markdown("Téléversez une image pour obtenir une prédiction via l'API FastAPI.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affiche l’image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    # Bouton pour lancer la prédiction
    if st.button("Envoyer à l'API pour prédiction"):
        BUTTON_CLICKS.inc() # Increment the button click counter
        try:
            with st.spinner("Envoi de l’image à l’API..."):
                response = requests.post(
                    url="http://fastapi-api:8000/predict",
                    files={"file": uploaded_file.getvalue()},
                )
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.success(f"Résultat : **{prediction}**")
                st.markdown(
                    '''
                    Consultez le tableau de bord [MLflow](http://localhost:5000)
                    pour suivre les expériences et les métriques.
                    ''',
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Erreur de l'API : {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Impossible de se connecter à l'API.\
                     Vérifiez qu'elle est bien lancée sur le port 8000."
            )
