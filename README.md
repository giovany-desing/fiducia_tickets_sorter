# 🎯 Fiducia Tickets Classifier API

El objetivo de este desarrollo es clasificar tickets en tres categorias deacuerdo a la descripcion o queja del usuario, se usan tecnicas de nlp para el procesamiento de texto y asi mismo machine learning para la clasificacion, una ves el ticket clasificado se guardara la categoria en una base de datos postgresl para luego con un tablero o dashboard validar KPIs 

## 📦 Despliegue Rápido

```bash
# 1. Clonar repositorio
git clone <url-del-repositorio>
cd fiducia_tickets_sorter

# 2. Desplegar con Docker
./deploy.sh

# 3. Verificar
curl http://localhost:5002/health


#Endpoints Disponibles

GET /health - Estado del servicio
POST /predict - Predicción individual
POST /predict/batch - Predicción en lote

# ARCHIVOS DEL PROYECTO

utils -->> Donde se guarda los archivos python con todas las utilizades necesarias para el  preprocesamiento de datos y conexion a postgresql y ejecucion de experimentos:
    connect_to_postgresql.py -->> archivo encargado de la conexion a la base de datos para insertar las predicciones.

    preprocessing.py -->> archivo encargado del preprocesamiento de datos.

    train_model.py -->> archivo encargado de ejecutar los experimentos para luego se trackeados con mlflow, el modelo con mejor desempeño se guardara en la carpeta models para luego ser cargado a la api.

models --> al ejecutar el archivo train_model.py el mejor modelo se guardara en esta carpeta para luego ser cargado para el funcionamiento de los enpoints de prediccion.

app.py -->> archivo de la apliacion aqui se ejecutan los endpoints de prediccion tanto del ticket individual como en batch