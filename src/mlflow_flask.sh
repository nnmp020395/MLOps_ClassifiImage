#!/bin/bash
echo "📡 Lancement de MLflow UI avec Flask pur sur http://127.0.0.1:5001"
export MLFLOW_SERVER_NO_GUNICORN=true
mlflow server --host 127.0.0.1 --port 5001
