"""
preprocess.py — Módulo de preprocesamiento para el pipeline de House Prices.

Pasos:
  1. Eliminar columnas con alta proporción de nulos (configurables en config.yaml)
  2. Separar features y variable objetivo
  3. Imputar nulos: mediana en numéricas, moda en categóricas
  4. Codificar variables categóricas con One-Hot Encoding
  5. Escalar variables numéricas con StandardScaler
  6. Dividir en conjuntos de entrenamiento y prueba

Puede ejecutarse directamente para validar el resultado del preprocesamiento:
    python src/preprocess.py
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path: str = "config.yaml") -> dict:
    """Carga la configuración desde config.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos cargados.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset no encontrado en: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def drop_high_null_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Elimina columnas con >80% de valores nulos definidas en config.yaml.

    Args:
        df: DataFrame original.
        drop_cols: Lista de columnas a eliminar.

    Returns:
        DataFrame sin las columnas especificadas.
    """
    before = df.shape[1]
    df = df.drop(columns=drop_cols, errors="ignore")
    dropped = before - df.shape[1]
    print(f"[INFO] Columnas eliminadas (alta proporción de nulos): {dropped}")
    return df


def impute_missing(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Imputa valores nulos:
    - Variables numéricas → mediana
    - Variables categóricas → moda (valor más frecuente)

    Args:
        df: DataFrame con nulos.
        num_cols: Lista de columnas numéricas.
        cat_cols: Lista de columnas categóricas.

    Returns:
        DataFrame sin valores nulos.
    """
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    remaining_nulls = df.isnull().sum().sum()
    print(f"[INFO] Nulos restantes después de imputación: {remaining_nulls}")
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> tuple[pd.DataFrame, list]:
    """
    Codifica variables categóricas usando One-Hot Encoding.
    Elimina la primera categoría (drop='first') para evitar multicolinealidad.

    Args:
        df: DataFrame con columnas categóricas.
        cat_cols: Lista de columnas categóricas a codificar.

    Returns:
        Tuple (DataFrame codificado, lista de nombres de columnas generadas).
    """
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols).tolist()
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = df.drop(columns=cat_cols).join(encoded_df)
    print(f"[INFO] One-Hot Encoding: {len(cat_cols)} columnas → {len(encoded_cols)} columnas generadas")
    return df, encoded_cols


def scale_numericals(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Escala variables numéricas usando StandardScaler (media=0, desviación=1).

    Args:
        df: DataFrame con columnas numéricas.
        num_cols: Lista de columnas numéricas a escalar.

    Returns:
        DataFrame con columnas numéricas escaladas.
    """
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[INFO] StandardScaler aplicado a {len(num_cols)} columnas numéricas")
    return df


def preprocess(df: pd.DataFrame, drop_cols: list, target_col: str,
               test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline completo de preprocesamiento:
        1. Eliminar columnas con alta proporción de nulos
        2. Separar X e y
        3. Imputar nulos
        4. One-Hot Encoding en categóricas
        5. StandardScaler en numéricas
        6. Split train/test

    Args:
        df: DataFrame crudo del CSV.
        drop_cols: Columnas a eliminar por alta proporción de nulos.
        target_col: Nombre de la variable objetivo.
        test_size: Proporción del conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Returns:
        X_train, X_test, y_train, y_test (todos como numpy arrays),
        feature_names (lista de nombres de columnas finales).
    """
    # 1. Eliminar columnas problemáticas
    df = drop_high_null_columns(df, drop_cols)

    # 2. Separar target y eliminar Id
    y = df[target_col].values
    X = df.drop(columns=[target_col, "Id"], errors="ignore")

    # 3. Identificar tipos de columnas
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"[INFO] Columnas numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    # 4. Imputar nulos
    X = impute_missing(X, num_cols, cat_cols)

    # 5. One-Hot Encoding en categóricas
    X, ohe_cols = encode_categoricals(X, cat_cols)

    # 6. StandardScaler en numéricas
    X = scale_numericals(X, num_cols)

    # 7. Split
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state
    )

    print(f"[INFO] Features finales: {len(feature_names)}")
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names


# ── Ejecución directa para validar el preprocesamiento ────────────────────────
if __name__ == "__main__":
    config = load_config("config.yaml")
    data_cfg = config["data"]

    df = load_data(data_cfg["path"])

    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df,
        drop_cols=data_cfg["drop_cols"],
        target_col=data_cfg["target_col"],
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )

    print("\n✅ Preprocesamiento completado exitosamente.")
    print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"   Nulos en X_train: {np.isnan(X_train).sum()}")
    print(f"   Nulos en X_test:  {np.isnan(X_test).sum()}")
    print(f"\n   Primeras 5 features: {feature_names[:5]}")
