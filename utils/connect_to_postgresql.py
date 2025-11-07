# modificar este script para supabase
import supabase
from datetime import datetime
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de Supabase
SUPABASE_URL = "https://hnvjgvigpjxxsjpbdmqk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhudmpndmlncGp4eHNqcGJkbXFrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1MTg2MDYsImV4cCI6MjA3ODA5NDYwNn0.PZNGkAAkB508f3gAsaf8c8Kyj8Xg8Hj5-xFXCWIfxD8"

def actualizar_causa_ticket(id_ticket, causa, conn_params=None):
    """
    Actualiza la causa de un ticket en Supabase.
    Mantiene la misma interfaz que la versión PostgreSQL local.
    """
    logger.info(f"Actualizando causa para ticket: {id_ticket} en Supabase")
    
    try:
        # 1. Conectar a Supabase (ignoramos conn_params para mantener compatibilidad)
        client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Conexión exitosa a Supabase")
        
        # 2. Ejecutar UPDATE en Supabase en la tabla tickets_fiducia
        update_data = {
            "causa": causa,
            "updated_at": datetime.now().isoformat()  # Agregar timestamp de actualización
        }
        
        # Ejecutar el UPDATE en la tabla tickets_fiducia
        response = client.table("tickets_fiducia").update(update_data).eq("number", id_ticket).execute()
        
        # Verificar si se actualizó algún registro
        if response.data:
            filas_afectadas = len(response.data)
            logger.info(f"UPDATE ejecutado en Supabase. Filas afectadas: {filas_afectadas}")
            
            return {
                'success': True,
                'message': f'Causa actualizada para ticket {id_ticket} en Supabase',
                'filas_afectadas': filas_afectadas,
                'data': response.data
            }
        else:
            logger.warning(f"No se encontró el ticket {id_ticket} para actualizar")
            return {
                'success': False,
                'error': f'Ticket {id_ticket} no encontrado en Supabase',
                'filas_afectadas': 0
            }
        
    except Exception as e:
        logger.error(f"Error actualizando en Supabase: {e}")
        return {
            'success': False, 
            'error': str(e),
            'filas_afectadas': 0
        }

def verificar_conexion_supabase():
    """Función adicional para verificar la conexión a Supabase"""
    try:
        client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        # Intentar una consulta simple en tickets_fiducia
        response = client.table("tickets_fiducia").select("number").limit(1).execute()
        logger.info("Conexión a Supabase verificada exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error verificando conexión a Supabase: {e}")
        return False

def verificar_columna_causa():
    """Verificar si la columna 'causa' existe en la tabla tickets_fiducia"""
    try:
        client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        # Intentar consultar la columna causa
        response = client.table("tickets_fiducia").select("causa").limit(1).execute()
        logger.info("Columna 'causa' existe en tickets_fiducia")
        return True
    except Exception as e:
        logger.error(f"La columna 'causa' no existe en tickets_fiducia: {e}")
        return False

# Parámetros de conexión (se mantienen por compatibilidad pero no se usan)
conn_params = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'mysecretpassword',
    'port': '5432'
}

# Ejemplo de uso (compatible con el código existente)
if __name__ == "__main__":
    # Verificar conexión primero
    if verificar_conexion_supabase():
        # Verificar si existe la columna causa
        if verificar_columna_causa():
            # Ejemplo de actualización
            resultado = actualizar_causa_ticket(
                id_ticket='INC1353571',  # Usando un ticket que sabemos existe
                causa=1
            )
            print("Resultado:", resultado)
        else:
            print("Error: La columna 'causa' no existe en la tabla tickets_fiducia")
            print("💡 Ejecuta este SQL en Supabase para crear la columna:")
            print("ALTER TABLE tickets_fiducia ADD COLUMN causa INTEGER;")
            print("ALTER TABLE tickets_fiducia ADD COLUMN updated_at TIMESTAMPTZ;")
    else:
        print("Error: No se pudo conectar a Supabase")