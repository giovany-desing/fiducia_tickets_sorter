import psycopg2
from datetime import datetime
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def actualizar_causa_ticket(id_ticket, causa, conn_params=None):
    logger.info(f"Actualizando causa para ticket: {id_ticket}")
    
    if conn_params is None:
        conn_params = {
            'host': 'localhost',
            'database': 'postgres',
            'user': 'postgres',
            'password': 'mysecretpassword',
            'port': '5432'
        }
    
    try:
        # 1. Conectar a la base de datos
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        logger.info("Conexión exitosa a PostgreSQL")
        
        
        # 2. Ejecutar UPDATE directo
        update_query = "UPDATE ada SET causa = %s WHERE number = %s;"
        cursor.execute(update_query, (causa, id_ticket))
        filas_afectadas = cursor.rowcount
        
        # 3. Confirmar cambios
        conn.commit()
        logger.info(f"UPDATE ejecutado. Filas afectadas: {filas_afectadas}")
        
        # 4. Cerrar conexión
        cursor.close()
        conn.close()
        logger.info("Conexión cerrada")
        
        return {
            'success': True,
            'message': f'Causa actualizada para ticket {id_ticket}',
            'filas_afectadas': filas_afectadas
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
            if 'cursor' in locals():
                cursor.close()
            conn.close()
        return {'success': False, 'error': str(e)}

# Parámetros de conexión
conn_params = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'mysecretpassword',
    'port': '5432'
}

# Ejemplo de uso
if __name__ == "__main__":
    resultado = actualizar_causa_ticket(
        id_ticket='INC1386412',
        causa=1
    )
    print("Resultado:", resultado)