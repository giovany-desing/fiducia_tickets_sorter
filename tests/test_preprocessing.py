import sys
import os
import pandas as pd
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS

import pytest

from utils.preprocessing import clean, limpiar_y_stem, save_dataset, load_config

def test_clean_function():
    # Caso 1: Texto con puntuación, números, URL y etiquetas HTML
    text_with_noise = "VINCULACION PREVENTAS FIDUCIARIA (ONE ID) - Se genera la vinculación de 1 cliente"
    # El resultado esperado debe coincidir con la salida real de la función
    expected_result = "vinculacion preventas fiduciaria one id  se genera la vinculacin de  cliente"
    assert clean(text_with_noise) == expected_result

    # Caso 2: Texto solo con espacios extras
    text_with_spaces = "25/07/2023 0602000139972626 - 51906728 Al cliente no le llega correo con documentos a firmar (ONE ID): Se solicita se envie el el  nit al correo  luzmarleny55"
    expected_result_spaces = "al cliente no le llega correo con documentos a firmar one id se solicita se envie el el  nit al correo"
    assert clean(text_with_spaces) == expected_result_spaces


def test_limpiar_y_stem_function():
    # Caso 1: Texto con stopwords, mayúsculas y palabras comunes
    text_to_process = "cerrado por sistemas sin respuesta"
    # El resultado esperado debe coincidir con la salida real de la función
    expected_result = "cerr sistem"
    assert limpiar_y_stem(text_to_process) == expected_result

    # Caso 2: Texto con solo stopwords
    text_only_stopwords = "no se obtuvo respuesta del usuario cerrado por sistemas"
    expected_result_stopwords = "obtuv usuari cerr sistem"
    assert limpiar_y_stem(text_only_stopwords) == expected_result_stopwords

# Funcion para crear un dataset de pruebas
@pytest.fixture
def dummy_df():

    return pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

# Crea una carpeta temporal para simular processed_path. tmp_path es un directorio temporal que pytest limpia solo.
@pytest.fixture
def temp_processed_path(tmp_path):
    return tmp_path

def test_save_dataset_creates_file(dummy_df, temp_processed_path, monkeypatch):
    # Simula el config.yaml
    config = {
        "data": {
            "processed_path": str(temp_processed_path)  # se usa carpeta temporal
        }
    }


    monkeypatch.setattr("utils.preprocessing.os.path.dirname",
                        lambda path: os.getcwd())

    output_file = save_dataset(dummy_df, filename="test.csv", config=config)

    # Verifica que el archivo se creó
    assert os.path.exists(output_file)

    # Lee el CSV y compara
    saved_df = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(dummy_df, saved_df)

