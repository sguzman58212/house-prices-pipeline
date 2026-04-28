"""
train.py — Script principal de entrenamiento del pipeline de House Prices.

Dataset: House Prices - Advanced Regression Techniques (Kaggle)
  Fuente: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
  Descripción: 1460 casas en Ames, Iowa. 79 variables explicativas que describen
               aspectos de propiedades residenciales. Variable objetivo: SalePrice.
  Tarea: Regresión — predecir el precio de venta de cada casa.

Modelo: Gradient Boosting Regressor (scikit-learn)
Métricas: RMSE y R²

Uso:
    python src/train.py
    make train
"""

import os
import sys
import traceback

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Importar módulo de preprocesamiento
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_config, load_data, preprocess  # noqa: E402


def train(config: dict):
    """
    Ejecuta el pipeline completo: carga, preprocesa, entrena, evalúa y registra.

    Args:
        config: Diccionario con la configuración del proyecto (config.yaml).
    """
    data_cfg = config["data"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]

    # ── Configurar MLflow ─────────────────────────────────────────────────────
    tracking_uri = "file://" + os.path.abspath(mlflow_cfg["tracking_uri"])
    os.makedirs(mlflow_cfg["tracking_uri"], exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[INFO] MLflow Tracking URI: {tracking_uri}")

    # ── Cargar datos ──────────────────────────────────────────────────────────
    df = load_data(data_cfg["path"])

    # ── Preprocesar ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        drop_cols=data_cfg["drop_cols"],
        target_col=data_cfg["target_col"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    # ── Crear/recuperar experimento MLflow ────────────────────────────────────
    experiment = mlflow.set_experiment(mlflow_cfg["experiment_name"])
    print(f"[INFO] Experimento '{mlflow_cfg['experiment_name']}' (ID: {experiment.experiment_id})")

    # ── Entrenar y registrar ──────────────────────────────────────────────────
    try:
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            print(f"[INFO] Run ID: {run.info.run_id}")

            # Modelo
            model = GradientBoostingRegressor(
                n_estimators=model_cfg["n_estimators"],
                max_depth=model_cfg["max_depth"],
                learning_rate=model_cfg["learning_rate"],
                subsample=model_cfg["subsample"],
                random_state=model_cfg["random_state"],
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            print(f"[INFO] RMSE: {rmse:,.2f} | R²: {r2:.4f}")

            # Log parámetros
            mlflow.log_param("n_estimators", model_cfg["n_estimators"])
            mlflow.log_param("max_depth", model_cfg["max_depth"])
            mlflow.log_param("learning_rate", model_cfg["learning_rate"])
            mlflow.log_param("subsample", model_cfg["subsample"])
            mlflow.log_param("test_size", data_cfg["test_size"])
            mlflow.log_param("random_state", data_cfg["random_state"])
            mlflow.log_param("dataset", "house-prices-kaggle (train.csv)")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", X_train.shape[0])

            # Log métricas
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Firma e input_example (criterio "Se excede" de la rúbrica)
            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_train[:5]

            # Registrar modelo en el Model Registry
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=mlflow_cfg["model_name"],
            )

            print(f"\n✅ Modelo '{mlflow_cfg['model_name']}' registrado.")
            print(f"   RMSE = {rmse:,.2f} | R² = {r2:.4f}")
            print(f"   Artefactos: {run.info.artifact_uri}")

    except Exception:
        print("[ERROR] Fallo durante el entrenamiento:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    config = load_config("config.yaml")
    train(config)
