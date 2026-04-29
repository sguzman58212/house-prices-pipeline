# House Prices ML Pipeline

Pipeline reproducible de Machine Learning para predecir el precio de venta de casas en Ames, Iowa. Automatizado con GitHub Actions y tracking de experimentos con MLflow.

## Resultados del modelo

| Metrica | Valor |
|---------|-------|
| Modelo | Gradient Boosting Regressor |
| RMSE | ~$26,476 |
| R2 | ~0.909 |
| Features finales | 233 |
| Split | 80% entrenamiento / 20% prueba |

El modelo explica el 90.9% de la varianza del precio de venta con un error promedio de $26,476.

## Dataset

House Prices - Advanced Regression Techniques (Kaggle)
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

1.460 casas con 79 variables explicativas sobre propiedades residenciales en Ames, Iowa. Variable objetivo: SalePrice (precio de venta en USD). Problema de regresion supervisada.

## Requisitos

- Python 3.10
- pip
- make
- MLflow
- scikit-learn

## Instalacion

Clona el repositorio:

```bash
git clone https://github.com/sguzman58212/house-prices-pipeline.git
cd house-prices-pipeline
```

Descarga `train.csv` desde Kaggle y colócalo en la carpeta `data/`:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Instala las dependencias:

```bash
make install
```

## Uso

Preprocesar datos (opcional, para validar cada paso):

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

Verificar estilo del código:

```bash
make lint
```

Pipeline completo (install + lint + train + test):

```bash
make all
```

## Estructura del proyecto

```
house-prices-pipeline/
├── src/
│   ├── preprocess.py       # Modulo de preprocesamiento independiente
│   └── train.py            # Script principal de entrenamiento
├── tests/
│   └── test_pipeline.py    # 8 tests de validacion del pipeline
├── data/
│   └── train.csv           # Dataset de Kaggle (no incluido en el repo)
├── .github/
│   └── workflows/
│       └── ml.yml          # Pipeline CI/CD con GitHub Actions
├── config.yaml             # Hiperparametros y rutas configurables
├── Makefile                # Automatizacion de tareas
├── requirements.txt        # Dependencias
├── .gitignore
└── README.md
```

## Preprocesamiento

El modulo `src/preprocess.py` aplica los siguientes pasos:

1. Eliminacion de columnas con mas del 80% de nulos: PoolQC, MiscFeature, Alley, Fence, MasVnrType
2. Imputacion: mediana para numericas, moda para categoricas
3. One-Hot Encoding en 38 variables categoricas (233 features finales)
4. StandardScaler en 36 variables numericas
5. Train/Test split 80/20 con random_state=42

## Configuracion de hiperparametros

Los hiperparametros se controlan desde `config.yaml` sin modificar el codigo:

```yaml
model:
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  random_state: 42
```

## MLflow Tracking

Cada ejecucion de entrenamiento registra en MLflow:

- **Parametros**: n_estimators, max_depth, learning_rate, subsample, test_size, random_state, n_features, n_train_samples, dataset
- **Metricas**: rmse, r2
- **Modelo**: serializado con firma (signature) e input_example
- **Model Registry**: registrado como `house-prices-gbr`

Para visualizar los experimentos:

```bash
mlflow ui
```

Luego abre http://localhost:5000 en el navegador.

### Evidencia del modelo registrado

El modelo queda registrado en el MLflow Model Registry bajo el nombre `house-prices-gbr`. Cada run del experimento `house-prices-experiment` muestra:

- RMSE: 26476.28
- R2: 0.9086

## CI/CD con GitHub Actions

El workflow `.github/workflows/ml.yml` corre automaticamente en cada push a `main`:

1. Checkout del codigo
2. Configuracion de Python 3.10
3. `make install` — instalacion de dependencias
4. `make lint` — verificacion de estilo con flake8
5. Tests pre-entrenamiento
6. `make train` — entrenamiento completo del modelo
7. Tests completos incluyendo validacion de mlruns
8. Guarda `mlruns/` como artefacto del workflow por 7 dias

## Tests

El archivo `tests/test_pipeline.py` contiene 8 tests automatizados:

- `test_dataset_exists` — el CSV existe en la ruta configurada
- `test_dataset_not_empty` — el dataset no esta vacio
- `test_dataset_has_target` — la columna SalePrice existe
- `test_dataset_shape` — minimo 1000 filas y 50 columnas
- `test_target_positive` — todos los precios son positivos
- `test_preprocess_output` — X e y son arrays numpy sin valores nulos
- `test_preprocess_removes_drop_cols` — columnas eliminadas no aparecen en X
- `test_mlruns_exists_after_train` — MLflow genero los artefactos

Todos los tests pasan en local y en GitHub Actions.

## Autor

Sebastian Guzman Gomez
Maestria en Ciencia de Datos — Universidad EAN
Modulo 3: MLOps y Analitica en la Nube
