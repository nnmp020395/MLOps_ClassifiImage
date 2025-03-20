import os
import sqlite3
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Chemin de la base de données SQLite
DB_PATH = "../mlops_images.db"

# Dossier parent où les images sont stockées
BASE_SAVE_PATH = "../dataset"

def update_s3_paths(db_path):
    """
    Met à jour la colonne `url_s3` avec le chemin local des images téléchargées.

    Args:
        db_path (str): Chemin du fichier SQLite.
    """
    logging.info("Début de la mise à jour des chemins locaux dans `url_s3`.")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Connexion à SQLite réussie.")

        # Récupérer les URLs source et labels depuis la base
        cursor.execute("SELECT id, url_source, label FROM plants_data")
        images = cursor.fetchall()

        updated_count = 0

        for image_id, url_source, label in images:
            filename = url_source.split("/")[-1]
            local_path = os.path.abspath(os.path.join(BASE_SAVE_PATH, label, filename))

            # Vérifier si le fichier existe avant de mettre à jour
            if os.path.exists(local_path):
                cursor.execute("UPDATE plants_data SET url_s3 = ? WHERE id = ?", (local_path, image_id))
                updated_count += 1
            else:
                logging.warning(f"Image non trouvée pour mise à jour : {local_path}")

        conn.commit()
        logging.info(f"{updated_count} enregistrements mis à jour avec l'URL locale.")

    except sqlite3.Error as e:
        logging.error(f"Erreur SQLite lors de la mise à jour : {e}")
        if conn:
            conn.rollback()

    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        logging.info("Connexion SQLite fermée.")
    
    logging.info("Mise à jour terminée.")
