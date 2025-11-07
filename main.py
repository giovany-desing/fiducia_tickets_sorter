from flask import Flask, request, jsonify
import os
import sys
import pandas as pd
import logging
from datetime import datetime
import traceback
import json
import joblib
from pathlib import Path

# Configurar rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

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

# Función para cargar el modelo (COMPATIBLE CON DOCKER Y LOCAL)
def load_model():
    global model
    try:
        # ✅ DETECTAR AUTOMÁTICAMENTE SI ESTAMOS EN DOCKER O LOCAL
        current_file = Path(__file__).resolve()
        
        # Opción 1: Si estamos en Docker (/app/app.py)
        if str(current_file).startswith('/app/'):
            models_dir = Path("/app/models")
            logger.info("🔧 Entorno detectado: DOCKER")
        
        # Opción 2: Si estamos en local (desarrollo)
        else:
            # Buscar la carpeta del proyecto (donde está app.py)
            project_root = current_file.parent
            models_dir = project_root / "models"
            logger.info("🔧 Entorno detectado: LOCAL")
            logger.info(f"📁 Ruta del proyecto: {project_root}")
        
        latest_model_path = models_dir / "latest_model.pkl"
        latest_metadata_path = models_dir / "latest_model_metadata.json"
        
        logger.info(f"🔍 Buscando modelo en: {latest_model_path}")
        
        # Verificar que existen los archivos
        if not latest_model_path.exists():
            logger.error(f"❌ No se encuentra el archivo del modelo: {latest_model_path}")
            
            # Listar archivos disponibles en models/ para debugging
            if models_dir.exists():
                available_files = list(models_dir.iterdir())
                logger.info("Archivos disponibles en models/:")
                for file in available_files:
                    logger.info(f"  - {file}")
            else:
                logger.error("❌ La carpeta models/ no existe")
            
            return False
        
        if not latest_metadata_path.exists():
            logger.warning("⚠️ No se encuentra el archivo de metadata del modelo")
        
        # Cargar el modelo usando joblib
        model = joblib.load(latest_model_path)
        logger.info("✅ Modelo cargado exitosamente")
        
        # Cargar metadata si está disponible
        if latest_metadata_path.exists():
            with open(latest_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"📊 Modelo: {metadata.get('model_name', 'Desconocido')}")
            logger.info(f"🎯 F1-Score: {metadata.get('f1_score', 'Desconocido')}")
        else:
            logger.info("ℹ️ No hay metadata disponible para este modelo")
        
        # Verificación adicional del modelo
        logger.info(f"🔍 Tipo del modelo cargado: {type(model)}")
        
        # Verificar que el modelo tiene los métodos necesarios
        if hasattr(model, 'predict'):
            logger.info("✅ El modelo tiene método 'predict'")
        else:
            logger.error("❌ El modelo no tiene método 'predict'")
            return False
            
        if hasattr(model, 'predict_proba'):
            logger.info("✅ El modelo tiene método 'predict_proba'")
        else:
            logger.warning("⚠️ El modelo no tiene método 'predict_proba'")
        
        # Hacer una prueba rápida de predicción con datos dummy
        try:
            # Crear datos de prueba (texto vacío o simple)
            test_text = ["test prediction"]
            prediction = model.predict(test_text)
            logger.info(f"🧪 Prueba de predicción exitosa. Shape: {prediction.shape}")
            logger.info("✅ Modelo listo para predicciones en producción.")
        except Exception as test_e:
            logger.warning(f"⚠️ La prueba de predicción falló: {test_e}")
            logger.info("ℹ️ El modelo se cargó pero puede haber problemas con las predicciones")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {e}")
        logger.error("🔍 Traceback completo:", exc_info=True)
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
        required_fields = ['short_description', 'close_notes', 'ticket_id']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos faltantes: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Extraer datos
        short_description = data['short_description']
        close_notes = data['close_notes']
        ticket_id = data['ticket_id']
        
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
        
        # Resumen final
        logger.info("📊 RESUMEN FINAL DEL PROCESAMIENTO BATCH")
        logger.info(f"📦 Total de tickets recibidos: {len(tickets)}")
        logger.info(f"✅ Tickets procesados exitosamente: {len([p for p in predictions if 'prediction' in p])}")
        logger.info(f"❌ Tickets con errores: {errores_procesamiento}")
        logger.info(f"💾 Actualizaciones BD exitosas: {actualizaciones_exitosas}")
        logger.info(f"⚠️ Actualizaciones BD fallidas: {actualizaciones_fallidas}")
        
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
        # ✅ Puerto configurable por variable de entorno
        port = int(os.environ.get('PORT', 5002))
        logger.info(f"🚀 Iniciando servidor Flask en puerto {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("No se pudo cargar el modelo. Terminando aplicación.")
        sys.exit(1)