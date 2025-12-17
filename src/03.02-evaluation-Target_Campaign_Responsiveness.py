import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score)

# Leer ficheros de Test 
X_test = pd.read_csv('..\\data\\test\\Target_Campaign_Responsiveness\\X_test.csv', index_col=0) 
y_test = pd.read_csv('..\\data\\test\\Target_Campaign_Responsiveness\\y_test.csv', index_col=0).squeeze() # Squeeze nos convierte el dataframe y_test a un numpy Series

# Leer modelo entrenado:
nombre_archivo = '..\\models\\modelo_xgboost-classifier_Target_Campaign_Responsiveness.pkl'
if os.path.exists(nombre_archivo):
    print(f"\nCargando el modelo desde: {nombre_archivo}...")

    # Inicializamos una variable para el modelo cargado
    modelo_cargado = None
    try:
        # Abrimos el archivo en modo lectura binaria ('rb')
        with open(nombre_archivo, 'rb') as archivo:
            # Usamos pickle.load() para leer el objeto modelo del archivo
            modelo_cargado = pickle.load(archivo)
        print("¡Modelo cargado exitosamente!")

    except Exception as e:
        print(f"Ocurrió un error al cargar: {e}")

    # Verificar métricas del modelo:
    y_pred = modelo_cargado.predict(X_test)

    # Predicción de probabilidades (para ROC AUC)
    y_proba = modelo_cargado.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)      # Importante: Capacidad de encontrar los 1s
    prec = precision_score(y_test, y_pred)  # Importante: Cuántos de los que dijimos 1 son realmente 1
    f1 = f1_score(y_test, y_pred)

    results = {
        'ROC AUC': roc, 
        'Recall (Sensibilidad)': rec, 
        'Precision': prec,
        'F1 Score': f1,
        'Accuracy': acc
    }

    # Imprimir métricas
    print("___________________________________________")
    print(" ---------- Métricas del modelo: ----------")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value}")
    print("___________________________________________")

else:
    print(f"\nEl archivo {nombre_archivo} no se encontró. No se pudo cargar.")
