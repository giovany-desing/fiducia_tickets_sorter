from flask import Flask, request, jsonify
import os
import sys
import mlflow
import pandas as pd
import logging
from datetime import datetime
import traceback

# Configurar rutas
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utils.preprocessing import clean, limpiar_y_stem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variable global para el modelo
model = None

# funcion para cargar el modelo desde Mlflow
def load_model():
    global model
    try:
        # Configurar MLflow para cargar el modelo
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        # Asegurar que estamos en la carpeta raiz del proyecto
        if os.path.basename(project_root) != "fiducia_tickets_sorter":
            # Buscar la carpeta correcta
            base_dir = os.path.dirname(current_file_path)
            while os.path.basename(base_dir) != "fiducia_tickets_sorter" and base_dir != "/":
                base_dir = os.path.dirname(base_dir)
            if os.path.basename(base_dir) == "fiducia_tickets_sorter":
                project_root = base_dir
        
        tracking_dir = os.path.join(project_root, "mlruns")
        mlflow.set_tracking_uri(f"file://{tracking_dir}")
        
        # Cargar el modelo XGBoost 
        model_name = "XGBoost_pipeline"
        model_version = "1"
        model_uri = f"models:/{model_name}/{model_version}"
        
        logger.info(f"Intentando cargar modelo: {model_uri}")
        logger.info(f"Desde directorio: {tracking_dir}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Modelo '{model_name}' versión {model_version} cargado exitosamente.")
        
        # Verificación adicional del modelo
        logger.info(f"Tipo del modelo cargado: {type(model)}")
        logger.info("Modelo listo para predicciones.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {e}")
        
        # Información de debug más detallada
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            logger.info("=== DEBUG: Modelos registrados ===")
            models = client.search_registered_models()
            if models:
                for rm in models:
                    logger.info(f"Modelo: {rm.name}")
                    versions = client.get_latest_versions(rm.name)
                    for v in versions:
                        logger.info(f"  - Versión {v.version} (Estado: {v.status})")
            else:
                logger.info("No se encontraron modelos registrados")
            logger.info("===================================")
            
        except Exception as debug_e:
            logger.error(f"Error en debug: {debug_e}")
            
        return False

def preprocess_text(short_description, close_notes):
    """Preprocesa el texto de la misma forma que en entrenamiento"""
    try:
        # Crear DataFrame temporal
        temp_df = pd.DataFrame({
            'clean_short_description_stem': [short_description],
            'clean_close_notes_stem': [close_notes]
        })
        
        # Combinar texto
        combined_text = temp_df["clean_short_description_stem"] + " " + temp_df["clean_close_notes_stem"]
        
        # Aplicar preprocessing
        processed_text = combined_text.apply(clean).apply(limpiar_y_stem)
        
        return processed_text
        
    except Exception as e:
        logger.error(f"Error en preprocessing: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para hacer predicciones"""
    try:
        # Verificar que el modelo esté cargado
        if model is None:
            return jsonify({
                'error': 'Modelo no disponible. Contacte al administrador.'
            }), 500
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No se enviaron datos. Formato esperado: JSON'
            }), 400
        
        # Validar campos requeridos
        required_fields = ['short_description', 'close_notes']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos faltantes: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Preprocesar texto
        short_description = data['short_description']
        close_notes = data['close_notes']
        
        processed_text = preprocess_text(short_description, close_notes)
        
        # Hacer predicción
        prediction = model.predict(processed_text)
        logger.info(f"ESTA ES LA PREDICCION: {prediction}")
        # Preparar respuesta
        response = {
            'prediction': int(prediction[0]),
            'input': {
                'short_description': short_description,
                'close_notes': close_notes,
                'processed_text': processed_text.iloc[0]
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"Predicción realizada: {prediction[0]}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': 'Error interno del servidor',
            'message': str(e),
            'status': 'error'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Endpoint para predicciones en lote"""
    try:
        if model is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        data = request.get_json()
        
        if not data or 'tickets' not in data:
            return jsonify({
                'error': 'Formato incorrecto. Esperado: {"tickets": [{"short_description": "...", "close_notes": "..."}]}'
            }), 400
        
        tickets = data['tickets']
        
        if not isinstance(tickets, list) or len(tickets) == 0:
            return jsonify({'error': 'Lista de tickets vacía o formato incorrecto'}), 400
        
        predictions = []
        
        for i, ticket in enumerate(tickets):
            try:
                if 'short_description' not in ticket or 'close_notes' not in ticket:
                    predictions.append({
                        'index': i,
                        'error': 'Campos faltantes: short_description, close_notes'
                    })
                    continue
                
                processed_text = preprocess_text(ticket['short_description'], ticket['close_notes'])
                prediction = model.predict(processed_text)
                
                predictions.append({
                    'index': i,
                    'prediction': int(prediction[0]),
                    'input': ticket
                })
                
            except Exception as e:
                predictions.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total_processed': len(tickets),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Endpoint de información general"""
    return jsonify({
        'message': 'API de Predicción de Tickets ML',
        'version': '1.0',
        'endpoints': {
            'health': '/health - Verificar estado de la API',
            'predict': '/predict - Predicción individual (POST)',
            'batch': '/predict/batch - Predicciones en lote (POST)'
        },
        'example_request': {
            'short_description': 'se solicita reenviar documentos al correo',
            'close_notes': 'se realiza reenvio de documento'
        }
    })

if __name__ == '__main__':
    # Cargar modelo al inicializar
    if load_model():
        logger.info("Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=5003, debug=False)
    else:
        logger.error("No se pudo cargar el modelo. Terminando aplicación.")
        sys.exit(1)