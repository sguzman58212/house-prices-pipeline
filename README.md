# House Prices ML Pipeline

Pipeline de Machine Learning para predecir el precio de venta de casas en Ames, Iowa. Automatizado con GitHub Actions y tracking de experimentos con MLflow.

## Dataset

House Prices - Advanced Regression Techniques (Kaggle)
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

1.460 casas con 79 variables explicativas sobre características de propiedades residenciales en Ames, Iowa. La variable objetivo es SalePrice (precio de venta en USD). Es un problema de regresión.

## Resultados

Modelo: Gradient Boosting Regressor
RMSE: ~$26,476
R²: ~0.909
Features finales: 233 (tras preprocesamiento)
Split: 80% entrenamiento / 20% prueba

## Instalación

Clona el repositorio y descarga train.csv desde Kaggle, colócalo en la carpeta data/.

```bash
git clone https://github.com/TU_USUARIO/house-prices-pipeline.git
cd house-prices-pipeline
make install
```

## Uso

Preprocesar datos (opcional, para validar):

```bash
python src/preprocess.py
```

Entrenar el modelo:

```bash
make train
```

Ejecutar tests:

```bash
make test
```

Pipeline completo:

```bash
make all
```

## Preprocesamiento

El módulo src/preprocess.py aplica los siguientes pasos:

1. Eliminación de columnas con más del 80% de nulos: PoolQC, MiscFeature, Alley, Fence, MasVnrType
2. Imputación: mediana para numéricas, moda para categóricas
3. One-Hot Encoding en 38 variables categóricas
4. StandardScaler en 36 variables numéricas
5. Train/Test split con random_state=42

## MLflow

Los experimentos se guardan localmente en mlruns/. Para visualizarlos:

```bash
mlflow ui
```

Cada ejecución registra parámetros, métricas (rmse, r2), firma del modelo, input_example y el modelo en el Model Registry bajo el nombre house-prices-gbr.

## Configuración

Los hiperparámetros y rutas se controlan desde config.yaml sin modificar el código.

## CI/CD

El workflow .github/workflows/ml.yml corre en cada push a main: instala dependencias, lint, tests, entrenamiento y guarda mlruns/ como artefacto del workflow.

## Autor

Sebastián Guzmán Gómez
Maestría en Ciencia de Datos — Universidad EAN
Módulo 3: MLOps y Analítica en la Nube
