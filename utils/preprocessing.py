import re
import string
import nltk
import yaml
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS

def load_config(config_path='../config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def clean(text):
    # Convertir a minúsculas
    text = str(text).lower()

    # Eliminar textos entre corchetes (ej.: etiquetas)
    text = re.sub(r'\[.*?\]', '', text)

    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Eliminar etiquetas HTML
    text = re.sub(r'<.*?>+', '', text)

    # Eliminar signos de puntuación
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Eliminar saltos de línea
    text = re.sub(r'\n', ' ', text)

    # Eliminar palabras que contienen números
    text = re.sub(r'\w*\d\w*', '', text)

    # Eliminar emojis y caracteres especiales (no ASCII)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Eliminar espacios extras al inicio y final
    text = text.strip()

    return text

def limpiar_y_stem(texto):
    stopwords_es = set(stopwords.words('spanish'))
    stemmer = SnowballStemmer("spanish")

    stopwords_personalizadas = {
    "buen", "buenos", "dia", "tardes","buenas", "grupo", "asignación", "asignacin", "asignanar" ,"asignado", "davinci", "preventa", "fiduciaria",
    "respuesta", "proveedor", "favor", "pronta", "quedo", "atento",
    "sebastian", "sebastin", "prado", "vanegas", "identificación", "one", "id", "vinculacion", "preventas", "muchas", "gracias", "cliente",
    "atentos", "soporte", "cc"
    }
    stopwords_totales = stopwords_es.union(stopwords_personalizadas)
        # Tokenizar en palabras
    tokens = word_tokenize(texto, language="spanish")

        # Filtrar stopwords y palabras no alfabéticas
    tokens_filtrados = [
            t for t in tokens
            if t.isalpha() and t.lower() not in stopwords_totales
        ]

        # Aplicar stemming
    tokens_stem = [stemmer.stem(t.lower()) for t in tokens_filtrados]

    # Volver a unir en un solo string
    return " ".join(tokens_stem)

