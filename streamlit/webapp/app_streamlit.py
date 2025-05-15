"""
Application Streamlit pour la classification d'images avec DINOv2.
"""

import io
import logging
import os
import threading
import time
from io import BytesIO
from uuid import uuid4

import boto3
import botocore
import requests
from PIL import Image
from prometheus_client import REGISTRY, Counter, start_http_server

import streamlit as st

# ------------------ LOGGING SETUP ------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/streamlit_app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

lock = threading.Lock()


# ------------------ PROMETHEUS METRICS ------------------
def setup_metrics():
    """
    Démarre le serveur de métriques Prometheus et initialise les compteurs
    pour les visites de page et les clics sur le bouton de prédiction.

    Returns:
        tuple: Compteurs pour les visites de page et les clics sur le bouton.
    """
    try:
        page_views = REGISTRY._names_to_collectors["streamlit_page_views"]
    except KeyError:
        page_views = Counter(
            "streamlit_page_views", "Nombre de visites de la page Streamlit"
        )
    try:
        button_clicks = REGISTRY._names_to_collectors["predict_button_clicks"]
    except KeyError:
        button_clicks = Counter(
            "predict_button_clicks", "Nombre de clics sur le bouton de prédiction"
        )
    return page_views, button_clicks

def start_metrics_server():
    """
    Démarre le serveur de métriques Prometheus sur le port 8502.
    """
    start_http_server(8502)
    while True:
        time.sleep(10)


if "metrics_started" not in st.session_state:
    st.session_state.PAGE_VIEWS, st.session_state.BUTTON_CLICKS = setup_metrics()
    threading.Thread(target=start_metrics_server, daemon=True).start()
    st.session_state.metrics_started = True

PAGE_VIEWS = st.session_state.PAGE_VIEWS
BUTTON_CLICKS = st.session_state.BUTTON_CLICKS

# ------------------ CONFIGURATION MINIO ------------------
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
minio_access_key = os.getenv("AWS_ACCESS_KEY_ID")
minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = "image-dandelion-grass"
region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ------------------ INTERFACE STREAMLIT ------------------
st.set_page_config(page_title="Prédiction DINOv2", layout="centered")
PAGE_VIEWS.inc()
page = st.sidebar.selectbox("Navigation", ["Public", "Admin"])

if page == "Public":
    st.title("Classificateur Dandelion vs Grass (DINOv2)")
    st.markdown("Téléversez une image pour obtenir une prédiction via l'API FastAPI.")
    uploaded_file = st.file_uploader(
        "Choisir une image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image sélectionnée", use_column_width=True)
        except Exception as e:
            logger.error(f"Erreur chargement image : {e}")
            st.error("L'image ne peut pas être lue.")
            st.stop()

        if st.button("Envoyer à l'API pour prédiction"):
            BUTTON_CLICKS.inc()
            try:
                with st.spinner("Envoi de l’image à l’API..."):
                    response = requests.post(
                        url="http://fastapi-api:8000/predict",
                        files={"file": uploaded_file.getvalue()},
                    )
                if response.status_code == 200:
                    prediction = response.json().get("prediction")
                    st.success(f"Résultat : **{prediction}**")
                    logger.info(f"Prédiction : {prediction}")

                    # Vérification des doublons après prédiction
                    with st.spinner("Vérification des doublons..."):
                        duplicate_response = requests.post(
                            url="http://fastapi-api:8000/check_duplicate",
                            files={"file": uploaded_file.getvalue()},
                            data={"label": prediction},
                        )

                        if duplicate_response.status_code == 200:
                            result = duplicate_response.json()
                            status = result.get("status")
                            message = result.get("message")
                            st.info("À bientôt !")
                            logger.info(f"Doublon : {status} - {message}")
                        else:
                            st.error(f"Erreur API doublon : {duplicate_response.text}")

                    st.markdown(
                        """
                        Consultez les dashboards :
                        - [MLflow](http://localhost:5001)
                        - [Grafana](http://localhost:3000)
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Erreur API : {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connexion à l'API impossible.")

# -------------------- ADMIN INTERFACE ------------------
elif page == "Admin":
    st.title("Interface Admin : Validation des prédictions")
    password = st.text_input("Mot de passe admin :", type="password")
    if password != st.secrets["Admin"]["password"]:
        st.warning("Mot de passe incorrect ou manquant.")
        st.stop()

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name=region_name,
        config=botocore.config.Config(connect_timeout=5, read_timeout=30),
        use_ssl=False,
    )

    @st.cache_data(show_spinner=False)
    def get_image_from_minio(bucket: str, key: str) -> Image.Image:
        """
        Récupère une image depuis MinIO.
        """
        try:
            buf = BytesIO()
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception as e:
            st.error(f"Erreur image {key} : {e}")
            return None

    def list_new_images():
        """
        Récupère la liste des images à valider depuis MinIO.
        """
        try:
            response = s3.list_objects_v2(
                Bucket=bucket_name, Prefix="raw/new_data/pending_validation/"
            )
            return [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].endswith((".jpg", ".png"))
            ]
        except Exception as e:
            st.error(f"Erreur MinIO : {e}")
            return []

    image_keys = list_new_images()
    if not image_keys:
        st.success("Aucune image à valider.")
        st.stop()

    for key in image_keys:
        st.markdown("---")
        cols = st.columns([1.5, 1, 0.7, 0.7])
        action_key = f"action_{key}"
        if action_key not in st.session_state:
            st.session_state[action_key] = False

        with cols[0]:
            image = get_image_from_minio(bucket_name, key)
            if image:
                st.image(image, caption=key, use_column_width=True)
            else:
                continue

        predicted_label = key.split("_")[-1].split(".")[0]
        with cols[1]:
            st.markdown("**Label prédit :**")
            st.markdown(f"`{predicted_label}`")
        with cols[2]:
            validated = st.checkbox("✅", key=f"ok_{key}")
        with cols[3]:
            error = st.checkbox("❌", key=f"err_{key}")

        if not st.session_state[action_key]:
            with lock:
                if validated and not error:
                    new_filename = f"{uuid4().hex}_{predicted_label}.jpg"
                    validated_key = f"raw/new_data/corrected_data/{new_filename}"

                    buf = BytesIO()
                    s3.download_fileobj(bucket_name, key, buf)
                    buf.seek(0)
                    image = Image.open(buf).convert("RGB")
                    if image.size[0] > 1024 or image.size[1] > 1024:
                        image.thumbnail((780, 780))
                        st.warning(f"Redimensionnée : {image.size}")
                    upload_buf = BytesIO()
                    image.save(upload_buf, format="JPEG", quality=80)
                    upload_buf.seek(0)

                    s3.upload_fileobj(upload_buf, bucket_name, validated_key)
                    time.sleep(1)
                    s3.delete_object(Bucket=bucket_name, Key=key)
                    logger.info(f"Validée : {key} -> {validated_key}")
                    st.session_state[action_key] = True
                    st.experimental_rerun()

                elif error and not validated:
                    corrected_label = (
                        "grass" if predicted_label == "dandelion" else "dandelion"
                    )
                    new_filename = f"{uuid4().hex}_{corrected_label}.jpg"
                    corrected_key = f"raw/new_data/corrected_data/{new_filename}"

                    buf = BytesIO()
                    s3.download_fileobj(bucket_name, key, buf)
                    buf.seek(0)
                    image = Image.open(buf).convert("RGB")
                    if image.size[0] > 1024 or image.size[1] > 1024:
                        image.thumbnail((1024, 1024))
                        logger.info(f"Redimensionnée : {image.size}")
                    upload_buf = BytesIO()
                    image.save(upload_buf, format="JPEG", quality=90)
                    upload_buf.seek(0)

                    s3.upload_fileobj(upload_buf, bucket_name, corrected_key)
                    time.sleep(1)
                    s3.delete_object(Bucket=bucket_name, Key=key)
                    logger.warning(f"Corrigée : {key} -> {corrected_key}")
                    st.session_state[action_key] = True
                    st.experimental_rerun()

                elif validated and error:
                    st.error(
                        "Une image ne peut pas être validée **et** marquée en erreur."
                    )
