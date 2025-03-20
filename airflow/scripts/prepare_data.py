import os
import sqlite3
import logging
from PIL import Image

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Chemin de la base de données SQLite
DB_PATH = "../mlops_images.db"

# Dossier où les images sont stockées
BASE_SAVE_PATH = "../dataset"

# Dossier de sortie des données nettoyées
OUTPUT_DIR = "../clean_dataset"

def check_image_validity(image_path):
    """
    Vérifie si une image est valide (non corrompue).

    Args:
        image_path (str): Chemin de l'image.

    Returns:
        bool: True si l'image est valide, False sinon.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Vérifie l'intégrité
        return True
    except Exception:
        return False

def clean_data(db_path):
    """
    Vérifie et nettoie les données avant l'entraînement :
    - Vérifie l'existence et la validité des images.
    - Supprime les images corrompues ou inaccessibles.
    - Prépare les dossiers pour PyTorch.

    Args:
        db_path (str): Chemin du fichier SQLite.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Connexion à SQLite réussie.")

        # Récupérer toutes les images avec leurs chemins locaux
        cursor.execute("SELECT id, url_s3, label FROM plants_data WHERE url_s3 IS NOT NULL")
        images = cursor.fetchall()

        valid_count = 0
        invalid_count = 0

        # Organisation des fichiers nettoyés
        for image_id, image_path, label in images:
            if not os.path.exists(image_path) or not check_image_validity(image_path):
                logging.warning(f"Image corrompue ou introuvable : {image_path}, suppression de la DB.")
                cursor.execute("DELETE FROM plants_data WHERE id = ?", (image_id,))
                invalid_count += 1
                continue

            # Dossier propre pour l'entraînement (format ImageFolder de PyTorch)
            output_label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(output_label_dir, exist_ok=True)

            # Nouveau chemin de stockage
            filename = os.path.basename(image_path)
            new_path = os.path.join(output_label_dir, filename)

            # Évite les doublons
            if not os.path.exists(new_path):
                os.rename(image_path, new_path)
                cursor.execute("UPDATE plants_data SET url_s3 = ? WHERE id = ?", (new_path, image_id))
                valid_count += 1
            else:
                logging.info(f"Doublon détecté, suppression : {image_path}")
                os.remove(image_path)
                cursor.execute("DELETE FROM plants_data WHERE id = ?", (image_id,))
                invalid_count += 1

        conn.commit()
        logging.info(f"{valid_count} images valides et nettoyées.")
        logging.info(f"{invalid_count} images corrompues ou dupliquées supprimées.")

    except sqlite3.Error as e:
        logging.error(f"Erreur SQLite : {e}")
        if conn:
            conn.rollback()

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        logging.info("Connexion SQLite fermée.")

if __name__ == "__main__":
    clean_data(DB_PATH)