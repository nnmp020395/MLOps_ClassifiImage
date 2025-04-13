"""
Module pour insérer des URLs d'images dans une base SQLite.

Ce fichier contient une fonction utilisée dans un pipeline Airflow pour :
1. Créer une table `plants_data` si elle n'existe pas.
2. Insérer des URLs d'images brutes dans la base SQLite.
"""
import logging

from sqlalchemy import create_engine, text

# ------------------ CONFIGURATION DE LA CONNEXION SQL ------------------

# Connexion à une base SQLite locale via SQLAlchemy
# (Remplace par une URI MySQL si nécessaire pour la prod)
sql_alchemy_conn = "sqlite:////Users/fabreindira/airflow/airflow.db"
engine = create_engine(sql_alchemy_conn)


def url_to_sql(**kwargs):
    """
    Fonction utilisée dans le pipeline Airflow pour insérer des URLs d'images \
        dans une base SQLite.

    Étapes :
    1. Crée la table `plants_data` si elle n'existe pas déjà.
    2. Génère des URLs d'images brutes (dandelion / grass) hébergées sur GitHub.
    3. Insère ces URLs dans la table, sans doublons (`INSERT OR IGNORE`).

    Paramètres :
    ----------
    **kwargs : dict
        Paramètres passés automatiquement par Airflow, non utilisés ici directement.

    Retourne :
    ---------
    int
        Nombre d'enregistrements insérés (même si les doublons sont ignorés).
    """
    with engine.connect() as connection:
        try:
            # Création de la table si elle n'existe pas
            CREATE_TABLE_QUERY = """
            CREATE TABLE IF NOT EXISTS plants_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url_source VARCHAR(255) NOT NULL UNIQUE,
                url_s3 VARCHAR(255) DEFAULT NULL,
                label VARCHAR(50) NOT NULL
            );
            """
            connection.execute(text(CREATE_TABLE_QUERY))
            logging.info("Table `plants_data` vérifiée/créée.")

            # Construction des URLs GitHub pour les deux classes
            base_url = (
                "https://raw.githubusercontent.com/btphan95/greenr-airflow/"
                "refs/heads/master/data"
            )
            labels = ["dandelion", "grass"]
            data_to_insert = [
                {
                    "url_source": f"{base_url}/{label}/{i:08d}.jpg",
                    "url_s3": None,
                    "label": label,
                }
                for label in labels
                for i in range(200)
            ]

            # Insertion des données (avec duplication ignorée)
            INSERT_QUERY = """
            INSERT OR IGNORE INTO plants_data (url_source, url_s3, label)
            VALUES (:url_source, :url_s3, :label)
            """

            with connection.begin():
                connection.execute(text(INSERT_QUERY), data_to_insert)

            logging.info(f"{len(data_to_insert)} enregistrements insérés avec succès.")
            return len(data_to_insert)

        except Exception as e:
            logging.exception(f"Erreur MySQL lors de l'insertion des données: {e}")
            connection.rollback()
            return 0
