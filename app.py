from flask import Flask, request, jsonify
import os
import sys
import mlflow
import pandas as pd
import logging
from datetime import datetime
import traceback




# Configurar rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# CORREGIR: El archivo se llama connect_to_postgresql.py (con QL)
from utils.preprocessing import clean, limpiar_y_stem
from utils.connect_to_postgresql import actualizar_causa_ticket

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
    """Endpoint principal para hacer predicciones y actualizar causa en BD"""
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
        required_fields = ['short_description', 'close_notes', 'ticket_id']  # ✅ Agregado ticket_id
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos faltantes: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Extraer datos
        short_description = data['short_description']
        close_notes = data['close_notes']
        ticket_id = data['ticket_id']  # ✅ Nuevo campo requerido
        
        # Preprocesar texto
        processed_text = preprocess_text(short_description, close_notes)
        
        # Hacer predicción
        prediction = model.predict(processed_text)
        prediction_value = int(prediction[0])
        
        logger.info(f"PREDICCIÓN REALIZADA: {prediction_value} para ticket {ticket_id}")
        
        # ✅ ACTUALIZAR CAUSA EN LA BASE DE DATOS
        
        # Mapear la predicción a una causa legible
        causas = {
            0: 'Davibox sin documentos',
            1: 'Reenvio de documentos', 
            2: 'Errores generales',
        }
        
        causa_prediccion = causas.get(prediction_value, f'Categoría {prediction_value}')
        
        # Actualizar la base de datos
        resultado_actualizacion = actualizar_causa_ticket(
            id_ticket=ticket_id,
            causa=causa_prediccion
        )
        
        # Preparar respuesta
        response = {
            'prediction': prediction_value,
            'causa_asignada': causa_prediccion,
            'actualizacion_bd': {
                'success': resultado_actualizacion['success'],
                'message': resultado_actualizacion.get('message', ''),
                'ticket_id': ticket_id
            },
            'input': {
                'ticket_id': ticket_id,
                'short_description': short_description,
                'close_notes': close_notes,
                'processed_text': processed_text.iloc[0]
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # Log del resultado
        if resultado_actualizacion['success']:
            logger.info(f"✅ Causa actualizada en BD para ticket {ticket_id}: {causa_prediccion}")
        else:
            logger.error(f"❌ Error actualizando causa para ticket {ticket_id}: {resultado_actualizacion.get('error')}")
        
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
    """Endpoint para predicciones en lote con actualización en BD"""
    try:
        logger.info("🎯 INICIANDO PREDICCIÓN BATCH")
        logger.info(f"📦 Request recibido - Content-Type: {request.content_type}")
        
        if model is None:
            logger.error("❌ Modelo no disponible para predicción batch")
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        data = request.get_json()
        logger.debug(f"📊 Datos recibidos: {len(data.get('tickets', [])) if data else 0} tickets")
        
        if not data or 'tickets' not in data:
            logger.warning("⚠️ Formato incorrecto en request batch")
            return jsonify({
                'error': 'Formato incorrecto. Esperado: {"tickets": [{"ticket_id": "...", "short_description": "...", "close_notes": "..."}]}'
            }), 400
        
        tickets = data['tickets']
        logger.info(f"📋 Procesando lote de {len(tickets)} tickets")
        
        if not isinstance(tickets, list) or len(tickets) == 0:
            logger.warning("⚠️ Lista de tickets vacía o formato incorrecto")
            return jsonify({'error': 'Lista de tickets vacía o formato incorrecto'}), 400
        
        # Mapeo de predicciones a causas
        causas = {
            0: 'Davibox sin documentos',
            1: 'Reenvio de documentos', 
            2: 'Errores generales',
        }
        logger.debug(f"🎯 Mapeo de causas configurado: {causas}")
        
        predictions = []
        actualizaciones_exitosas = 0
        actualizaciones_fallidas = 0
        errores_procesamiento = 0
        
        logger.info("🔄 Iniciando procesamiento de tickets...")
        
        for i, ticket in enumerate(tickets):
            ticket_log_prefix = f"[Ticket {i+1}/{len(tickets)}]"
            try:
                logger.debug(f"{ticket_log_prefix} Procesando ticket: {ticket.get('ticket_id', 'No ID')}")
                
                # Validar campos requeridos
                campos_requeridos = ['ticket_id', 'short_description', 'close_notes']
                campos_faltantes = [campo for campo in campos_requeridos if campo not in ticket]
                
                if campos_faltantes:
                    logger.warning(f"⚠️ {ticket_log_prefix} Campos faltantes: {campos_faltantes}")
                    predictions.append({
                        'index': i,
                        'error': f'Campos faltantes: {campos_faltantes}',
                        'ticket_id': ticket.get('ticket_id', 'No proporcionado')
                    })
                    errores_procesamiento += 1
                    continue
                
                # Extraer datos
                ticket_id = ticket['ticket_id']
                short_description = ticket['short_description'][:100] + "..." if len(ticket['short_description']) > 100 else ticket['short_description']
                close_notes = ticket['close_notes'][:100] + "..." if len(ticket['close_notes']) > 100 else ticket['close_notes']
                
                logger.info(f"🔍 {ticket_log_prefix} Ticket ID: {ticket_id}")
                logger.debug(f"   Descripción: {short_description}")
                logger.debug(f"   Notas: {close_notes}")
                
                # Preprocesar y predecir
                logger.debug(f"{ticket_log_prefix} Preprocesando texto...")
                processed_text = preprocess_text(ticket['short_description'], ticket['close_notes'])
                logger.debug(f"{ticket_log_prefix} Texto preprocesado: {len(processed_text)} caracteres")
                
                logger.debug(f"{ticket_log_prefix} Realizando predicción...")
                prediction = model.predict(processed_text)
                prediction_value = int(prediction[0])
                causa_prediccion = causas.get(prediction_value, f'Categoría {prediction_value}')
                
                logger.info(f"🎯 {ticket_log_prefix} Predicción: {prediction_value} -> {causa_prediccion}")
                
                # Actualizar base de datos
                logger.debug(f"{ticket_log_prefix} Actualizando base de datos...")
                resultado_actualizacion = actualizar_causa_ticket(
                    id_ticket=ticket_id,
                    causa=causa_prediccion
                )
                
                # Contar actualizaciones
                if resultado_actualizacion['success']:
                    actualizaciones_exitosas += 1
                    logger.info(f"✅ {ticket_log_prefix} BD actualizada exitosamente")
                else:
                    actualizaciones_fallidas += 1
                    logger.error(f"❌ {ticket_log_prefix} Error en BD: {resultado_actualizacion.get('error')}")
                
                # Agregar a resultados
                prediction_data = {
                    'index': i,
                    'ticket_id': ticket_id,
                    'prediction': prediction_value,
                    'causa_asignada': causa_prediccion,
                    'actualizacion_bd': {
                        'success': resultado_actualizacion['success'],
                        'message': resultado_actualizacion.get('message', ''),
                        'error': resultado_actualizacion.get('error', '')
                    }
                }
                
                # Solo incluir input detallado si está en modo debug
                if logger.level <= logging.DEBUG:
                    prediction_data['input'] = {
                        'short_description': ticket['short_description'][:200] + "..." if len(ticket['short_description']) > 200 else ticket['short_description'],
                        'close_notes': ticket['close_notes'][:200] + "..." if len(ticket['close_notes']) > 200 else ticket['close_notes'],
                        'processed_text': processed_text.iloc[0] if hasattr(processed_text, 'iloc') else str(processed_text)[:200] + "..."
                    }
                
                predictions.append(prediction_data)
                logger.debug(f"✅ {ticket_log_prefix} Procesamiento completado")
                
            except Exception as e:
                errores_procesamiento += 1
                error_msg = f"Error procesando ticket: {str(e)}"
                logger.error(f"❌ {ticket_log_prefix} {error_msg}")
                logger.debug(f"   Traceback: {traceback.format_exc()}")
                
                predictions.append({
                    'index': i,
                    'ticket_id': ticket.get('ticket_id', 'No proporcionado'),
                    'error': error_msg
                })
        
        # Resumen final detallado
        logger.info("=" * 60)
        logger.info("📊 RESUMEN FINAL DEL PROCESAMIENTO BATCH")
        logger.info("=" * 60)
        logger.info(f"📦 Total de tickets recibidos: {len(tickets)}")
        logger.info(f"✅ Tickets procesados exitosamente: {len([p for p in predictions if 'prediction' in p])}")
        logger.info(f"❌ Tickets con errores: {errores_procesamiento}")
        logger.info(f"💾 Actualizaciones BD exitosas: {actualizaciones_exitosas}")
        logger.info(f"⚠️ Actualizaciones BD fallidas: {actualizaciones_fallidas}")
        logger.info(f"⏱️ Tiempo total: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        # Preparar respuesta
        response_data = {
            'predictions': predictions,
            'summary': {
                'total_tickets': len(tickets),
                'procesados_exitosamente': len([p for p in predictions if 'prediction' in p]),
                'con_errores': errores_procesamiento,
                'actualizaciones_exitosas': actualizaciones_exitosas,
                'actualizaciones_fallidas': actualizaciones_fallidas,
                'timestamp': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"📤 Enviando respuesta con {len(predictions)} predicciones")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error("💥 ERROR CRÍTICO EN PREDICCIÓN BATCH")
        logger.error(f"❌ Error: {e}")
        logger.error(f"🔍 Traceback completo: {traceback.format_exc()}")
        logger.error(f"📦 Request data: {request.get_data()[:500]}...")  # Log parcial del request
        
        return jsonify({
            'error': 'Error interno del servidor en procesamiento batch',
            'message': str(e),
            'status': 'error'
        }), 500

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
        app.run(host='0.0.0.0', port=5006, debug=False)
    else:
        logger.error("No se pudo cargar el modelo. Terminando aplicación.")
        sys.exit(1)