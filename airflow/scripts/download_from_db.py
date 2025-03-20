import os
import sqlite3
import requests
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Chemin de la base de données SQLite
DB_PATH = "../mlops_images.db"

# Dossier parent où stocker les images
BASE_SAVE_PATH = "../dataset"

def get_image_urls(db_path):
    """
    Récupère les URLs des images stockées dans la base SQLite.

    Args:
        db_path (str): Chemin du fichier SQLite.

    Returns:
        list: Liste de tuples (url_source, label).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Récupération des URLs et labels
        cursor.execute("SELECT url_source, label FROM plants_data")
        images = cursor.fetchall()
        
        conn.close()
        logging.info(f"{len(images)} images trouvées dans la base de données.")
        return images

    except sqlite3.Error as e:
        logging.error(f"Erreur SQLite: {e}")
        return []

def download_images(images):
    """
    Télécharge les images à partir des URLs et les stocke dans le bon dossier.

    Args:
        images (list): Liste de tuples (url_source, label).
    """
    for url, label in tqdm(images, desc="Téléchargement des images"):
        # Création du dossier correspondant au label
        save_dir = os.path.join(BASE_SAVE_PATH, label)
        os.makedirs(save_dir, exist_ok=True)
        
        # Nom de l'image basé sur son URL
        filename = url.split("/")[-1]
        save_path = os.path.join(save_dir, filename)

        # Télécharger l'image si elle n'existe pas déjà
        if not os.path.exists(save_path):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(save_path, "wb") as file:
                        file.write(response.content)
                    logging.info(f"Image téléchargée : {save_path}")
                else:
                    logging.warning(f"Erreur {response.status_code} lors du téléchargement de {url}")
            except Exception as e:
                logging.error(f"Erreur lors du téléchargement de {url}: {e}")
        else:
            logging.info(f"Image déjà existante : {save_path}")

if __name__ == "__main__":
    # Récupérer les URLs des images
    images = get_image_urls(DB_PATH)
    
    # Télécharger les images
    download_images(images)
