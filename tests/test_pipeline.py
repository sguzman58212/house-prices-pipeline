"""
Tests básicos de validación del pipeline de ML.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Asegurar que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import load_config, preprocess  # noqa: E402


@pytest.fixture
def config():
    """Carga la configuración del proyecto."""
    return load_config("config.yaml")


@pytest.fixture
def df(config):
    """Carga el dataset."""
    return pd.read_csv(config["data"]["path"])


def test_dataset_exists(config):
    """El dataset debe existir en la ruta configurada."""
    assert os.path.exists(config["data"]["path"]), (
        f"Dataset no encontrado en: {config['data']['path']}"
    )


def test_dataset_not_empty(df):
    """El dataset no debe estar vacío."""
    assert len(df) > 0, "El dataset está vacío"


def test_dataset_has_target(df, config):
    """El dataset debe contener la columna objetivo."""
    target = config["data"]["target_col"]
    assert target in df.columns, f"Columna objetivo '{target}' no encontrada"


def test_dataset_shape(df):
    """El dataset debe tener al menos 1000 filas y 50 columnas."""
    assert df.shape[0] >= 1000, f"Pocas filas: {df.shape[0]}"
    assert df.shape[1] >= 50, f"Pocas columnas: {df.shape[1]}"


def test_target_positive(df, config):
    """SalePrice debe ser siempre positivo."""
    target = config["data"]["target_col"]
    assert (df[target] > 0).all(), "Hay valores negativos o cero en SalePrice"


def test_preprocess_output(df, config):
    """El preprocesamiento debe devolver arrays numpy sin nulos."""
    data_cfg = config["data"]
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        drop_cols=data_cfg["drop_cols"],
        target_col=data_cfg["target_col"],
    )
    assert isinstance(X_train, np.ndarray), "X_train debe ser numpy array"
    assert isinstance(y_train, np.ndarray), "y_train debe ser numpy array"
    assert not np.isnan(X_train).any(), "X_train contiene NaN después del preprocesamiento"
    assert not np.isnan(X_test).any(), "X_test contiene NaN después del preprocesamiento"
    assert not np.isnan(y_train).any(), "y_train contiene NaN"
    assert len(feature_names) == X_train.shape[1], "Mismatch entre feature_names y columnas de X"


def test_preprocess_removes_drop_cols(df, config):
    """Las columnas con >80% nulos no deben aparecer en X."""
    data_cfg = config["data"]
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        drop_cols=data_cfg["drop_cols"],
        target_col=data_cfg["target_col"],
    )
    for col in data_cfg["drop_cols"]:
        assert col not in feature_names, f"Columna '{col}' no fue eliminada"


def test_mlruns_exists_after_train():
    """MLflow debe generar la carpeta mlruns después del entrenamiento."""
    assert os.path.exists("mlruns"), (
        "mlruns no existe — ejecuta 'make train' antes de los tests"
    )
