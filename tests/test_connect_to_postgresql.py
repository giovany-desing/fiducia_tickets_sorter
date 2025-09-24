import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# Importar las funciones del script original
import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.connect_to_postgresql import (
    conectar_base_datos,
    verificar_tabla_ada,
    verificar_version_postgres,
    verificar_ticket_existente,
    insertar_ticket_prueba,
    verificar_ticket_insertado,
    mostrar_tickets_existentes,
    cerrar_conexion,
    main
)

# Mock global para reutilizar en múltiples tests
def crear_mock_conexion():
    mock_conn = Mock()
    mock_cursor = Mock()
    return mock_conn, mock_cursor

# Tests para conectar_base_datos
def test_conectar_base_datos_exitosa():
   
    with patch('psycopg2.connect') as mock_connect:
        # Configurar mock
        mock_conn, mock_cursor = crear_mock_conexion()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Ejecutar función
        conn, cursor = conectar_base_datos()
        
        # Verificaciones
        assert conn == mock_conn
        assert cursor == mock_cursor
        mock_connect.assert_called_once()

def test_conectar_base_datos_error():
    """Test para error en conexión a la base de datos"""
    with patch('psycopg2.connect') as mock_connect:
        # Simular error
        mock_connect.side_effect = Exception("Error de conexión")
        
        # Ejecutar función
        conn, cursor = conectar_base_datos()
        
        # Verificaciones
        assert conn is None
        assert cursor is None

# Tests para verificar_tabla_ada
def test_verificar_tabla_ada_existente():
    """Test para verificar tabla ada existente"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (True,)
    
    resultado = verificar_tabla_ada(mock_cursor)
    
    assert resultado is True
    mock_cursor.execute.assert_called_once()

def test_verificar_tabla_ada_no_existente():
    """Test para verificar tabla ada no existente"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (False,)
    
    resultado = verificar_tabla_ada(mock_cursor)
    
    assert resultado is False

def test_verificar_tabla_ada_error():
    """Test para error al verificar tabla"""
    mock_cursor = Mock()
    mock_cursor.execute.side_effect = Exception("Error SQL")
    
    resultado = verificar_tabla_ada(mock_cursor)
    
    assert resultado is False

# Tests para verificar_version_postgres
def test_verificar_version_postgres():
    """Test para obtener versión de PostgreSQL"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = ("PostgreSQL 14.0",)
    
    resultado = verificar_version_postgres(mock_cursor)
    
    assert resultado == "PostgreSQL 14.0"
    mock_cursor.execute.assert_called_once_with("SELECT version();")

# Tests para verificar_ticket_existente
def test_verificar_ticket_existente_true():
    """Test para verificar ticket existente"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (1,)
    
    resultado = verificar_ticket_existente(mock_cursor, "TEST001")
    
    assert resultado is True
    mock_cursor.execute.assert_called_once_with(
        "SELECT COUNT(*) FROM ada WHERE number = %s;", ("TEST001",)
    )

def test_verificar_ticket_existente_false():
    """Test para verificar ticket no existente"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (0,)
    
    resultado = verificar_ticket_existente(mock_cursor, "TEST002")
    
    assert resultado is False

# Tests para insertar_ticket_prueba
@patch('datetime.datetime')
def test_insertar_ticket_prueba_exitoso(mock_datetime):
    """Test para inserción exitosa de ticket"""
    # Configurar mocks
    mock_now = datetime(2024, 1, 15, 10, 30, 0)
    mock_datetime.now.return_value = mock_now
    
    mock_conn, mock_cursor = crear_mock_conexion()
    
    # Ejecutar función
    resultado = insertar_ticket_prueba(mock_conn, mock_cursor)
    
    # Verificaciones
    assert resultado is True
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()

def test_insertar_ticket_prueba_error():
    """Test para error en inserción de ticket"""
    mock_conn, mock_cursor = crear_mock_conexion()
    mock_cursor.execute.side_effect = Exception("Error de inserción")
    
    resultado = insertar_ticket_prueba(mock_conn, mock_cursor)
    
    assert resultado is False
    mock_conn.rollback.assert_called_once()

# Tests para verificar_ticket_insertado
def test_verificar_ticket_insertado_true():
    """Test para verificar ticket insertado correctamente"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = ("TEST001", "closed", "Ticket de prueba")
    
    resultado = verificar_ticket_insertado(mock_cursor, "TEST001")
    
    assert resultado is True

def test_verificar_ticket_insertado_false():
    """Test para verificar ticket no insertado"""
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = None
    
    resultado = verificar_ticket_insertado(mock_cursor, "TEST001")
    
    assert resultado is False

# Tests para mostrar_tickets_existentes
def test_mostrar_tickets_existentes():
    """Test para mostrar tickets existentes"""
    mock_cursor = Mock()
    mock_tickets = [
        ("TICKET001", "closed", "Problema 1"),
        ("TICKET002", "open", "Problema 2")
    ]
    mock_cursor.fetchall.return_value = mock_tickets
    
    resultado = mostrar_tickets_existentes(mock_cursor, 2)
    
    assert resultado == mock_tickets
    mock_cursor.execute.assert_called_once_with(
        "SELECT number, state, short_description FROM ada LIMIT %s;", (2,)
    )

# Tests para cerrar_conexion
def test_cerrar_conexion_exitosa():
    """Test para cerrar conexión exitosamente"""
    mock_conn, mock_cursor = crear_mock_conexion()
    
    cerrar_conexion(mock_conn, mock_cursor)
    
    mock_cursor.close.assert_called_once()
    mock_conn.close.assert_called_once()

def test_cerrar_conexion_con_none():
    """Test para cerrar conexión con valores None"""
    # No debería lanzar excepción
    cerrar_conexion(None, None)

# Test para la función main
@patch('connect_to_postgresql.cerrar_conexion')
@patch('connect_to_postgresql.mostrar_tickets_existentes')
@patch('connect_to_postgresql.verificar_ticket_insertado')
@patch('connect_to_postgresql.insertar_ticket_prueba')
@patch('connect_to_postgresql.verificar_ticket_existente')
@patch('connect_to_postgresql.verificar_tabla_ada')
@patch('connect_to_postgresql.verificar_version_postgres')
@patch('connect_to_postgresql.conectar_base_datos')
def test_main_exitoso(mock_conectar, mock_version, mock_tabla, 
                     mock_verificar_existente, mock_insertar, 
                     mock_verificar_insert, mock_mostrar, mock_cerrar):
    """Test para la función main exitosa"""
    # Configurar mocks
    mock_conn, mock_cursor = crear_mock_conexion()
    mock_conectar.return_value = (mock_conn, mock_cursor)
    mock_tabla.return_value = True
    mock_verificar_existente.return_value = False
    mock_insertar.return_value = True
    mock_verificar_insert.return_value = True
    
    # Ejecutar main
    main()
    
    # Verificar que se llamaron todas las funciones en orden correcto
    mock_conectar.assert_called_once()
    mock_version.assert_called_once_with(mock_cursor)
    mock_tabla.assert_called_once_with(mock_cursor)
    mock_verificar_existente.assert_called_once_with(mock_cursor, 'TEST001')
    mock_insertar.assert_called_once_with(mock_conn, mock_cursor)
    mock_verificar_insert.assert_called_once_with(mock_cursor, 'TEST001')
    mock_mostrar.assert_called_once_with(mock_cursor, 5)
    mock_cerrar.assert_called_once_with(mock_conn, mock_cursor)

@patch('connect_to_postgresql.conectar_base_datos')
def test_main_conexion_fallida(mock_conectar):
    """Test para main con conexión fallida"""
    mock_conectar.return_value = (None, None)
    
    main()
    
    mock_conectar.assert_called_once()

# Tests de integración simples (opcionales)
def test_integracion_conexion_real():
    """Test de integración con base de datos real"""
    # Este test se salta si no hay base de datos disponible
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            database='postgres',
            user='postgres', 
            password='mysecretpassword',
            port='5432'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()
        conn.close()
    except Exception:
        pytest.skip("Base de datos no disponible")

# Fixture simple para reutilizar mocks
@pytest.fixture
def mock_database():
    """Fixture que proporciona mocks de base de datos"""
    mock_conn = Mock()
    mock_cursor = Mock()
    return mock_conn, mock_cursor

def test_con_mock_fixture(mock_database):
    """Test usando el fixture de mock"""
    mock_conn, mock_cursor = mock_database
    mock_cursor.fetchone.return_value = (True,)
    
    resultado = verificar_tabla_ada(mock_cursor)
    
    assert resultado is True