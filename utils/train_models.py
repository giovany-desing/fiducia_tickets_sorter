import os
import warnings
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import json
from datetime import datetime

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

# CONFIGURACIÓN DE LOGS
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

try:
    # CARGA Y LIMPIEZA DE DATOS
    logger.info("Cargando configuración del proyecto...")
    config = load_config()
    
    # ENCONTRAR CARPETA RAIZ DEL PROYECTO
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))  # Sube 2 niveles desde utils/
    
    # Verificar que estamos en la carpeta correcta
    expected_folder_name = "fiducia_tickets_sorter"
    if os.path.basename(project_root) != expected_folder_name:
        # Buscar la carpeta correcta
        for i in range(1, 4):
            potential_root = os.path.dirname(current_file_path)
            for _ in range(i):
                potential_root = os.path.dirname(potential_root)
            if os.path.basename(potential_root) == expected_folder_name:
                project_root = potential_root
                break
    
    logger.info("Raíz del proyecto identificada: %s", project_root)
    
    # CREAR CARPETAS NECESARIAS
    mlruns_dir = os.path.join(project_root, "mlruns")
    models_dir = os.path.join(project_root, "models")
    
    os.makedirs(mlruns_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    logger.info("Carpetas creadas: mlruns=%s, models=%s", mlruns_dir, models_dir)
    
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

    # CONFIGURACIÓN DE MLflow 
    tracking_dir = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    experiment_name = config["mlflow_tracking"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("Directorio mlruns: %s", tracking_dir)
    logger.info("Experimento: %s", experiment_name)
    logger.info("=============================")

    # MODELOS A EVALUAR
    modelos = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    # VARIABLES PARA GUARDAR EL MEJOR MODELO
    mejor_modelo = None
    mejor_nombre = None
    mejor_f1 = 0
    mejor_run_id = None
    resultados_modelos = {}

    # LOOP PARA EJECUTAR LOS EXPERIMENTOS
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

                # Guardar resultados para comparación
                resultados_modelos[nombre] = {
                    "accuracy": acc,
                    "f1_score": f1,
                    "precision": prec,
                    "recall": rec,
                    "run_id": run.info.run_id
                }

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

                # Guardar el pipeline completo en MLflow
                mlflow.sklearn.log_model(
                    sk_model=pipe,
                    artifact_path="model",
                    registered_model_name=f"{nombre}_pipeline"
                )

                # VERIFICAR SI ES EL MEJOR MODELO (basado en F1-score)
                if f1 > mejor_f1:
                    mejor_f1 = f1
                    mejor_modelo = pipe
                    mejor_nombre = nombre
                    mejor_run_id = run.info.run_id
                    logger.info("🏆 Nuevo mejor modelo: %s (F1: %.4f)", nombre, f1)

                logger.info("✅ Modelo %s registrado en MLflow. Run ID: %s", nombre, run.info.run_id)

            except Exception as e:
                logger.error("Error en modelo %s: %s", nombre, str(e))

    # GUARDAR EL MEJOR MODELO EN CARPETA MODELS
    if mejor_modelo is not None:
        logger.info("=" * 60)
        logger.info("🏆 GUARDANDO MEJOR MODELO")
        logger.info("=" * 60)
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_filename = f"best_model_{mejor_nombre}_{timestamp}.pkl"
        modelo_path = os.path.join(models_dir, modelo_filename)
        
        # Guardar el modelo usando joblib
        import joblib
        joblib.dump(mejor_modelo, modelo_path)
        logger.info("✅ Mejor modelo guardado en: %s", modelo_path)
        
        # Guardar metadata del mejor modelo
        metadata = {
            "model_name": mejor_nombre,
            "f1_score": mejor_f1,
            "run_id": mejor_run_id,
            "timestamp": timestamp,
            "model_path": modelo_path,
            "model_filename": modelo_filename
        }
        
        metadata_path = os.path.join(models_dir, f"best_model_metadata_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Metadata del modelo guardada en: %s", metadata_path)
        
        # También guardar un archivo "latest" para fácil acceso
        latest_model_path = os.path.join(models_dir, "latest_model.pkl")
        latest_metadata_path = os.path.join(models_dir, "latest_model_metadata.json")
        
        joblib.dump(mejor_modelo, latest_model_path)
        with open(latest_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Modelo 'latest' guardado para acceso rápido")
        
        # Guardar resumen de todos los modelos
        resultados_path = os.path.join(models_dir, f"model_comparison_{timestamp}.json")
        with open(resultados_path, 'w', encoding='utf-8') as f:
            json.dump(resultados_modelos, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Comparación de modelos guardada en: %s", resultados_path)
        
        # Mostrar resumen final
        logger.info("=" * 60)
        logger.info("🎯 RESUMEN FINAL - MEJOR MODELO")
        logger.info("=" * 60)
        logger.info("Modelo: %s", mejor_nombre)
        logger.info("F1-Score: %.4f", mejor_f1)
        logger.info("Run ID: %s", mejor_run_id)
        logger.info("Archivo: %s", modelo_filename)
        logger.info("=" * 60)

    else:
        logger.error("❌ No se pudo entrenar ningún modelo correctamente")

except Exception as e:
    logger.exception("Error crítico en la ejecución del script: %s", str(e)) 










