import psycopg2
from psycopg2 import sql
from datetime import datetime

def conectar_base_datos():
    """Establece conexión con la base de datos"""
    try:
        # Parámetros de conexión
        conn_params = {
            'host': 'localhost',
            'database': 'postgres',
            'user': 'postgres',
            'password': 'mysecretpassword',
            'port': '5432'
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        print("✅ Conexión exitosa!")
        return conn, cursor
        
    except Exception as e:
        print(f"❌ Error en conexión: {e}")
        return None, None

def verificar_tabla_ada(cursor):
    """Verifica si la tabla ada existe"""
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'ada'
            );
        """)
        tabla_existe = cursor.fetchone()[0]
        print(f"¿Tabla 'ada' existe? {tabla_existe}")
        return tabla_existe
        
    except Exception as e:
        print(f"❌ Error verificando tabla: {e}")
        return False

def verificar_version_postgres(cursor):
    """Obtiene la versión de PostgreSQL"""
    try:
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"Versión de PostgreSQL: {db_version[0]}")
        return db_version[0]
    except Exception as e:
        print(f"❌ Error obteniendo versión: {e}")
        return None

def verificar_ticket_existente(cursor, numero_ticket):
    """Verifica si un ticket ya existe"""
    try:
        cursor.execute("SELECT COUNT(*) FROM ada WHERE number = %s;", (numero_ticket,))
        existe_ticket = cursor.fetchone()[0]
        return existe_ticket > 0
    except Exception as e:
        print(f"❌ Error verificando ticket: {e}")
        return False

def insertar_ticket_prueba(conn, cursor):
    """Inserta un ticket de prueba en la tabla ada"""
    try:
        insert_query = """
            INSERT INTO ada (
                number, priority, state, close_code, closed_at, 
                close_notes, resolved_at, sys_created_on, calendar_stc,
                cmdb_ci, business_duration, service_offering, origin_id,
                caller_id, calendar_duration, description, short_description
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        
        # Datos de prueba
        ticket_prueba = (
            'TEST001',  # number
            '2 - High',  # priority
            'closed',  # state
            'Solved (Work Around)',  # close_code
            datetime.now(),  # closed_at
            'Ticket de prueba insertado desde Python. Resuelto satisfactoriamente.',  # close_notes
            datetime.now(),  # resolved_at
            datetime.now(),  # sys_created_on
            'Standard Change',  # calendar_stc
            'Server Principal',  # cmdb_ci
            120,  # business_duration (minutos)
            'IT Support',  # service_offering
            'WEB',  # origin_id
            'user123',  # caller_id
            180,  # calendar_duration (minutos)
            'Este es un ticket de prueba creado automáticamente para verificar la conexión a la base de datos y la funcionalidad de inserción.',  # description
            'Ticket de prueba - Conexión Python'  # short_description
        )
        
        cursor.execute(insert_query, ticket_prueba)
        conn.commit()
        print("✅ Insert de prueba ejecutado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error insertando ticket: {e}")
        conn.rollback()
        return False

def verificar_ticket_insertado(cursor, numero_ticket):
    """Verifica que el ticket se insertó correctamente"""
    try:
        cursor.execute("SELECT number, state, short_description FROM ada WHERE number = %s;", (numero_ticket,))
        ticket_insertado = cursor.fetchone()
        if ticket_insertado:
            print(f"📋 Ticket verificado: {ticket_insertado}")
            return True
        else:
            print("❌ Ticket no encontrado después del insert")
            return False
    except Exception as e:
        print(f"❌ Error verificando insert: {e}")
        return False

def mostrar_tickets_existentes(cursor, limite=5):
    """Muestra los tickets existentes en la tabla"""
    try:
        cursor.execute("SELECT number, state, short_description FROM ada LIMIT %s;", (limite,))
        tickets = cursor.fetchall()
        print(f"\n📊 Primeros {limite} tickets en la tabla:")
        for i, ticket in enumerate(tickets, 1):
            print(f"   {i}. {ticket[0]}: {ticket[2]} ({ticket[1]})")
        return tickets
    except Exception as e:
        print(f"❌ Error mostrando tickets: {e}")
        return []

def cerrar_conexion(conn, cursor):
    """Cierra la conexión a la base de datos"""
    try:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("🔌 Conexión cerrada")
    except Exception as e:
        print(f"❌ Error cerrando conexión: {e}")

# === PROGRAMA PRINCIPAL ===
def main():
    """Función principal que ejecuta todo el proceso"""
    conn, cursor = conectar_base_datos()
    
    if conn is None or cursor is None:
        return
    
    try:
        # 1. Verificar versión de PostgreSQL
        verificar_version_postgres(cursor)
        
        # 2. Verificar si la tabla ada existe
        tabla_existe = verificar_tabla_ada(cursor)
        
        if not tabla_existe:
            print("❌ La tabla 'ada' no existe. Créala primero en DBeaver.")
            return
        
        # 3. Verificar si ya existe el ticket de prueba
        numero_ticket_prueba = 'TEST001'
        ticket_existe = verificar_ticket_existente(cursor, numero_ticket_prueba)
        
        if not ticket_existe:
            # 4. Insertar ticket de prueba
            print("📝 Insertando nuevo ticket de prueba...")
            if insertar_ticket_prueba(conn, cursor):
                # 5. Verificar el insert
                verificar_ticket_insertado(cursor, numero_ticket_prueba)
        else:
            print("ℹ️  El ticket de prueba ya existe en la base de datos")
        
        # 6. Mostrar tickets existentes
        mostrar_tickets_existentes(cursor, 5)
        
    except Exception as e:
        print(f"❌ Error en el proceso principal: {e}")
    
    finally:
        # 7. Cerrar conexión
        cerrar_conexion(conn, cursor)

# Ejecutar el programa
if __name__ == "__main__":
    main()