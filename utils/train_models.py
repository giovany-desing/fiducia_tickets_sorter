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
import os
from preprocessing import clean, limpiar_y_stem, load_config, save_dataset

config = load_config()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(project_root, config['data']['raw_path'])

data = pd.read_csv(raw_path)

data["clean_short_description"] = data["short_description"].apply(clean)
data["clean_close_notes"] = data["close_notes"].apply(clean)

data["clean_short_description_stem"] = data["clean_short_description"].apply(limpiar_y_stem)
data["clean_close_notes_stem"] = data["clean_close_notes"].apply(limpiar_y_stem)


save_data = data[["clean_short_description_stem","clean_close_notes_stem"]]

saved_path = save_dataset(save_data, filename="dataset_processed.csv", config=config)
print("✅ Guardado en:", saved_path)


# Combina las dos columnas de texto en una sola
X = data['clean_short_description_stem'] + ' ' + data['clean_close_notes_stem']

# La columna de etiquetas (la variable objetivo o traget)
y = data['etiqueta']

# Divide los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




