import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import mlflow
import mlflow.sklearn 
import logging
import warnings
import yaml

from preprocessing import clean, limpiar_y_stem, load_config

#configuracion de logs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#data = pd.read_csv('../data_project/dataset_tickets.csv', encoding='latin-1')
config = load_config()
data = pd.read_csv(config['data']['raw_path'])


logging.info("DATA DE ENTRENAMIENTO CARGADA EXITOSAMENTE, INICIANDO CON PREPROCESAMIENTO.....")

data["clean_short_description"] = data["short_description"].apply(clean)
data["clean_close_notes"] = data["close_notes"].apply(clean)

data["clean_short_description_stem"] = data["clean_short_description"].apply(limpiar_y_stem)
data["clean_close_notes_stem"] = data["clean_close_notes"].apply(limpiar_y_stem)


# Combina las dos columnas de texto en una sola
X = data['clean_short_description_stem'] + ' ' + data['clean_close_notes_stem']

# La columna de etiquetas (la variable objetivo o traget)
y = data['etiqueta']

# Divide los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logging.info("PREPROCESAMIENTO TERMINADO")
logging.info("INICIANDO CON EL ENTRENAMIENTO DE MODELOS")



