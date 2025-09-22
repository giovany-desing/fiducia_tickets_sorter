import os
import sys
import mlflow
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("Root path:", root_path)  # Verifica que sea correcto

# Agregamos la carpeta raíz al sys.path
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Ahora debería funcionar la importación
from utils.preprocessing import clean, limpiar_y_stem

from utils.preprocessing import clean, limpiar_y_stem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURACIÓN DE MLflow
# ==========================================
# Ruta absoluta para el backend local (asegúrate de que sea la misma que en tu script de entrenamiento)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tracking_dir = os.path.join(project_root, "mlruns")
mlflow.set_tracking_uri(f"file://{tracking_dir}")

# ==========================================
# CARGAR EL MODELO REGISTRADO
# ==========================================
# El nombre del modelo es exactamente el que aparece en el Model Registry
model_name = "XGBoost_pipeline"
# La versión '1' es la que se registró automáticamente en tu primer run
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"

try:
    logger.info("Cargando el modelo...")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Modelo '{model_name}' versión {model_version} cargado exitosamente.")

    # ==========================================
    # PREPARAR NUEVOS DATOS PARA LA PREDICCIÓN
    # ==========================================
    # Estos datos deben tener el mismo formato que los que usaste para el entrenamiento
    # Es crucial que las columnas de texto ya estén preprocesadas
    nuevos_tickets = pd.DataFrame({
        'clean_short_description_stem': [
            "se solicita reenviar documentos al correo tatata@gmail.com"
        ],
        'clean_close_notes_stem': [
            "se realiza reenvio de documento nic"
        ]
    })
    
    # Combinar texto de la misma forma que en el script de entrenamiento
    X_new = nuevos_tickets["clean_short_description_stem"] + " " + nuevos_tickets["clean_close_notes_stem"]
    
    X_new = X_new.apply(clean)
    X_new = X_new.apply(limpiar_y_stem)



    # ==========================================
    # HACER PREDICCIONES
    # ==========================================
    logger.info("Haciendo predicciones...")
    predicciones = loaded_model.predict(X_new)

    print("\n--- Resultados de la Predicción ---")
    for i, pred in enumerate(predicciones):
        print(f"Texto de entrada: '{X_new.iloc[i]}'")
        print(f"Predicción del modelo: {pred}\n")
    
except Exception as e:
    logger.error(f"Error al cargar o predecir: {e}")

