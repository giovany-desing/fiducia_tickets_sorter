import os
import warnings
import logging
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from preprocessing import clean, limpiar_y_stem, load_config, save_dataset

# ==========================================
# CONFIGURACIÓN DE LOGS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

try:
    # ==========================================
    # CARGA Y LIMPIEZA DE DATOS
    # ==========================================
    logger.info("Cargando configuración del proyecto...")
    config = load_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, config["data"]["raw_path"])

    logger.info("Leyendo dataset desde: %s", raw_path)
    data = pd.read_csv(raw_path)
    logger.info("Dataset cargado con %d filas y %d columnas", data.shape[0], data.shape[1])

    logger.info("Aplicando limpieza de texto...")
    data["clean_short_description"] = data["short_description"].apply(clean)
    data["clean_close_notes"] = data["close_notes"].apply(clean)
    data["clean_short_description_stem"] = data["clean_short_description"].apply(limpiar_y_stem)
    data["clean_close_notes_stem"] = data["clean_close_notes"].apply(limpiar_y_stem)

    save_data = data[["clean_short_description_stem", "clean_close_notes_stem"]]
    saved_path = save_dataset(save_data, filename="dataset_processed.csv", config=config)
    logger.info("Dataset procesado guardado en: %s", saved_path)

    # Combinar texto
    logger.info("Combinando columnas de texto para el modelo...")
    X = data["clean_short_description_stem"] + " " + data["clean_close_notes_stem"]
    y = data["etiqueta"]

    logger.info("Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Split completado: Train=%d | Test=%d", X_train.shape[0], X_test.shape[0])

    # ==========================================
    # CONFIGURACIÓN DE MLflow
    # ==========================================
    # Ruta absoluta para el backend local
    tracking_dir = os.path.join(project_root, "mlruns")
    os.makedirs(tracking_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    experiment_name = config["mlflow_tracking"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    logger.info("Tracking URI MLflow: %s", mlflow.get_tracking_uri())
    logger.info("Experimento MLflow: %s", experiment_name)

    # ==========================================
    # MODELOS A EVALUAR
    # ==========================================
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    # ==========================================
    # LOOP DE EXPERIMENTOS
    # ==========================================
    for nombre, modelo in modelos.items():
        logger.info("=== Entrenando modelo: %s ===", nombre)

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", modelo),
        ])

        # cada run tiene su propio contexto
        with mlflow.start_run(run_name=nombre) as run:
            try:
                logger.info("Iniciando entrenamiento...")
                pipe.fit(X_train, y_train)
                logger.info("Entrenamiento finalizado.")

                y_pred = pipe.predict(X_test)

                # Métricas
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted")

                logger.info(
                    "Resultados %s -> Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
                    nombre, acc, f1, prec, rec
                )

                # Log de métricas
                mlflow.log_metrics({
                    "accuracy": acc,
                    "f1_weighted": f1,
                    "precision_weighted": prec,
                    "recall_weighted": rec,
                })

                # Parámetros del modelo
                mlflow.log_params({f"model__{k}": v for k, v in modelo.get_params().items()
                                   if isinstance(v, (int, float, str, bool, type(None)))})

                # Guardar el pipeline completo
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path="model",
                    registered_model_name=f"{nombre.replace(' ', '_')}_pipeline"
                )

                logger.info("✅ Modelo %s registrado en MLflow. Run ID: %s", nombre, run.info.run_id)

            except Exception as e:
                logger.error("Error en modelo %s: %s", nombre, str(e))

except Exception as e:
    logger.exception("Error crítico en la ejecución del script: %s", str(e))    










