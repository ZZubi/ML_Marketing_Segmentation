import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Leer ficheros de Test 
X_test = pd.read_csv('..\\data\\test\\Target_MntWineYear\\X_test.csv', index_col=0) 
y_test = pd.read_csv('..\\data\\test\\Target_MntWineYear\\y_test.csv', index_col=0).squeeze() # Squeeze nos convierte el dataframe y_test a un numpy Series

# Leer modelo entrenado:
nombre_archivo = '..\\models\\modelo_xgboost-regression_Target_MntWineYear.pkl'
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

    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_mae = mean_absolute_error(y_test, y_pred)
    final_r2_score = r2_score(y_test, y_pred)

    print("___________________________________________")
    print(" ---------- Métricas del modelo: ----------")
    print('RMSE Test:', final_rmse)
    print('MAE Test:', final_mae)
    print('R2 Score Test:', final_r2_score)
    print("___________________________________________")

else:
    print(f"\nEl archivo {nombre_archivo} no se encontró. No se pudo cargar.")
