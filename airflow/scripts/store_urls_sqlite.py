import sqlite3
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def url_to_sql(db_path):
    """
    Insère les URLs des images dans la base SQLite.

    Args:
        db_path (str): Chemin du fichier SQLite.

    Returns:
        int: Nombre d'enregistrements insérés.
    """
    try:
        # Connexion à la base de données SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Connexion à SQLite réussie.")

        # Création de la table si elle n'existe pas
        CREATE_TABLE_QUERY = """
        CREATE TABLE IF NOT EXISTS plants_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_source TEXT NOT NULL UNIQUE, 
            url_s3 TEXT DEFAULT NULL,
            label TEXT NOT NULL
        );
        """
        cursor.execute(CREATE_TABLE_QUERY)
        logging.info("Table `plants_data` vérifiée/créée.")

        # Génération dynamique des URLs
        base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
        labels = ["dandelion", "grass"]
        data_to_insert = [(f"{base_url}/{label}/{i:08d}.jpg", None, label) for label in labels for i in range(200)]

        # Insertion des données dans SQLite (évite les doublons)
        INSERT_QUERY = """
        INSERT OR IGNORE INTO plants_data (url_source, url_s3, label) 
        VALUES (?, ?, ?)
        """
        cursor.executemany(INSERT_QUERY, data_to_insert)
        conn.commit()
        inserted_rows = cursor.rowcount
        logging.info(f"{inserted_rows} enregistrements insérés avec succès.")

    except sqlite3.Error as e:
        logging.exception("Erreur SQLite lors de l'insertion des données")
        if conn:
            conn.rollback()
        return 0

    finally:
        # Fermeture de la connexion
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
        logging.info("Connexion SQLite fermée.")

    return inserted_rows


