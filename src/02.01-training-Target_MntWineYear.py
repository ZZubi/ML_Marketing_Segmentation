from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle

# Leer fichero procesado en paso "01-data_processing.py"
df = pd.read_csv('../data/processed/clean_data.csv') 

# Definir X (features) e y (target)
target_col = "Target_MntWinesYear"

X = df.drop(columns=[target_col, 'ID', 'Target_Campaign_Responsiveness']) # Eliminamos también el otro target y el 'ID'
y = df[target_col]

# Split de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=11,
    shuffle=True # Por si los datos llegaran con algún orden
)

# Modelo de regresión para la variable Target_MntWineYear
best_model_selected = Pipeline([
                        ('scaler', MinMaxScaler()), 
                        ('model', XGBRegressor(
                                    objective='reg:squarederror',
                                    random_state=42,
                                    n_estimators=210,
                                    learning_rate=0.08,
                                    max_depth=6,
                                    subsample=0.75
                                ))
                    ])

# Entrenar modelo
best_model_selected.fit(X_train, y_train)

# Guardar el modelo como fichero pickle:
nombre_archivo = '..\\models\\modelo_xgboost-regression_Target_MntWineYear.pkl'

# 2. Guardar (Serializar) el Modelo
# ----------------------------------
print(f"Guardando el modelo en: {nombre_archivo}...")

try:
    # Abrimos el archivo en modo escritura binaria ('wb')
    with open(nombre_archivo, 'wb') as archivo:
        # Usamos pickle.dump() para escribir el objeto modelo en el archivo
        pickle.dump(best_model_selected, archivo)
    print("¡Modelo guardado exitosamente!")
except Exception as e:
    print(f"Ocurrió un error al guardar: {e}")

# Guardar datos de train y test
# --------------------------------------------------------------------------
# X_train
file_path = Path('..\\data\\train\\Target_MntWineYear\\X_train.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)
X_train.to_csv(file_path)

# X_test
file_path = Path('..\\data\\test\\Target_MntWineYear\\X_test.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)
X_test.to_csv(file_path)

# y_train
file_path = Path('..\\data\\train\\Target_MntWineYear\\y_train.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)
y_train.to_frame().to_csv(file_path)

# y_test
file_path = Path('..\\data\\test\\Target_MntWineYear\\y_test.csv')
file_path.parent.mkdir(parents=True, exist_ok=True)
y_test.to_frame().to_csv(file_path)