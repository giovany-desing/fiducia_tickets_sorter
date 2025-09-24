import psycopg2
from psycopg2 import sql
from datetime import datetime

# Parámetros de conexión
conn_params = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'mysecretpassword',
    'port': '5432'
}

try:
    # Establecer conexión
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    # Ejecutar consulta de prueba
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print("✅ Conexión exitosa!")
    print(f"Versión de PostgreSQL: {db_version[0]}")
    
    # Verificar tabla ada
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'ada'
        );
    """)
    tabla_existe = cursor.fetchone()[0]
    print(f"¿Tabla 'ada' existe? {tabla_existe}")
    
    if tabla_existe:
        # Verificar si ya existe el ticket de prueba para no duplicar
        cursor.execute("SELECT COUNT(*) FROM ada WHERE number = 'TEST001';")
        existe_ticket = cursor.fetchone()[0]
        
        if existe_ticket == 0:
            # INSERT de prueba
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
            
            # Ejecutar el INSERT
            cursor.execute(insert_query, ticket_prueba)
            conn.commit()
            print("✅ Insert de prueba ejecutado correctamente")
            
            # Verificar el insert
            cursor.execute("SELECT number, state, short_description FROM ada WHERE number = 'TEST001';")
            ticket_insertado = cursor.fetchone()
            print(f"📋 Ticket verificado: {ticket_insertado}")
            
        else:
            print("ℹ️  El ticket de prueba ya existe en la base de datos")
            
            # Mostrar los tickets existentes
            cursor.execute("SELECT number, state, short_description FROM ada LIMIT 5;")
            tickets = cursor.fetchall()
            print("\n📊 Primeros 5 tickets en la tabla:")
            for ticket in tickets:
                print(f"   - {ticket[0]}: {ticket[2]} ({ticket[1]})")
    
    else:
        print("❌ La tabla 'ada' no existe. Creala primero en DBeaver.")
    
    # Cerrar conexión
    cursor.close()
    conn.close()
    print("🔌 Conexión cerrada")
    
except Exception as e:
    print(f"❌ Error: {e}")