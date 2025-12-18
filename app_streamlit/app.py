import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, roc_auc_score, classification_report)

# Configuración de la página
st.set_page_config(page_title="App de Predicción ML", layout="wide")
st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            <style>
            /* Estilo para el texto de las pestañas */
            button[data-baseweb="tab"] p {
                font-size: 24px;         /* Tamaño de la fuente */
                font-weight: bold;      /* Grosor de la fuente */
            }

            /* Estilo para aumentar el espacio (padding) de la pestaña */
            button[data-baseweb="tab"] {
                height: 60px;           /* Altura de la pestaña */
                width: 100%;            /* Opcional: ajustar ancho */
            }
            </style>
            """
st.markdown(st_style, unsafe_allow_html=True) # Esconde los elementos específicos de streamlit

# Constantes
MODEL_PATH_Target_MntWinesYear = '..\\models\\modelo_xgboost-regression_Target_MntWineYear.pkl'
MODEL_PATH_Target_Campaign_Responsiveness = '..\\models\\modelo_xgboost-classifier_Target_Campaign_Responsiveness.pkl'
CONSUMO_MEDIO_VINO_POR_CLIENTE = 152.64 

def cargar_modelo(path):
    """Carga el modelo desde un archivo pickle."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# --- Estructura de la App ---
st.title("🔮 Dashboard de Predicción")

# Definimos las pestañas (Tabs)
tab1, tab2 = st.tabs(["🍷 Target_MntWinesYear", "📣 Target_Campaign_Responsiveness"])

# --- PESTAÑA 1: Ejecución del Modelo ---
with tab1:
    st.header("Consumo de vino anual por cliente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Sube aquí las variables predictoras (X_test)")
        x_test_file = st.file_uploader("Cargar X_test (CSV)", type=['csv'])

    with col2:
        st.info("Sube aquí los valores reales del Target (y_test)")
        y_test_file = st.file_uploader("Cargar y_test (CSV)", type=['csv'])

    st.markdown("---")

    # Botón de ejecución
    if st.button("⚡ Ejecutar Modelo y Predecir", type="primary"):
        # Validaciones previas
        if (x_test_file is None) or (y_test_file is None):
            st.warning("⚠️ Por favor, sube los archivos X_test e y_test antes de continuar.")
        else:
            # 1. Cargar el modelo
            model = cargar_modelo(MODEL_PATH_Target_MntWinesYear)
            
            if model is None:
                st.error(f"❌ No se encontró el archivo '{MODEL_PATH_Target_MntWinesYear}' en el directorio.")
                st.markdown("**Consejo:** Asegúrate de que tu archivo .pkl esté en la misma carpeta o actualiza la variable `MODEL_PATH`.")
            else:
                try:
                    # 2. Leer los CSV
                    df_x = pd.read_csv(x_test_file, index_col=0)
                    df_y = pd.read_csv(y_test_file, index_col=0)

                    y_test = df_y["Target_MntWinesYear"]
                    
                    # 3. Realizar predicción
                    st.success("✅ Modelo cargado y datos leídos correctamente. Generando predicciones...")
                    predictions = model.predict(df_x)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    r2_score_metric = r2_score(y_test, predictions)
                    
                    # Métricas de la predicción
                    st.write("### Métricas de la predicción:")
                    st.write("##### MAE:", np.round(mae,2))
                    st.write("##### RMSE:", np.round(rmse,2))
                    st.write("##### R2 Score:", np.round(r2_score_metric,2))

                    # 4. Generar Gráfico de ejemplo (Matplotlib + Seaborn)
                    st.write("### Análisis Gráfico")

                    #####################################################################
                    # Mostramos un gráfico donde se visualice predicción vs dato real
                    # Crear un DataFrame para facilitar la visualización con seaborn
                    df_results = pd.DataFrame({'Valores Reales (Y_test)': y_test, 
                                            'Predicciones': predictions})

                    fig, ax = plt.subplots(figsize=(10, 10))

                    # 2. Scatter plot (usando ax)
                    ax.scatter(df_results['Valores Reales (Y_test)'], df_results['Predicciones'], 
                            alpha=0.6, label='Puntos (Predicción vs. Real)')

                    # 3. Línea de referencia Y=X
                    max_val = max(df_results['Valores Reales (Y_test)'].max(), df_results['Predicciones'].max())
                    min_val = min(df_results['Valores Reales (Y_test)'].min(), df_results['Predicciones'].min())
                    ax.plot([min_val, max_val], [min_val, max_val], 
                            color='red', linestyle='--', linewidth=2, label='Línea Y=X (Perfecta)')

                    # 4. Línea horizontal (axhline -> ax.axhline)
                    ax.axhline(y=CONSUMO_MEDIO_VINO_POR_CLIENTE, 
                            color='green', 
                            linestyle='-.', 
                            linewidth=1.5, 
                            label=f"Media $ vino cliente / año [{CONSUMO_MEDIO_VINO_POR_CLIENTE}]")

                    # 5. Línea vertical (axvline -> ax.axvline)
                    ax.axvline(x=CONSUMO_MEDIO_VINO_POR_CLIENTE, 
                            color='green', 
                            linestyle='-.', 
                            linewidth=1.5)

                    # 6. Configuración de etiquetas y estilo (set_title, set_xlabel...)
                    ax.set_title('Predicciones vs. Valores Reales')
                    ax.set_xlabel('Valores Reales (y_test)', fontsize=14)
                    ax.set_ylabel('Predicciones del Modelo (y_pred)', fontsize=14)
                    ax.grid(True)
                    ax.legend(facecolor='#f0f0f0')

                    #####################################################################

                    # Renderizar el gráfico en Streamlit
                    st.pyplot(fig)

                    # Mostrar los resultados
                    st.write("### Resultados de la predicción:")
                    df_results = df_x.copy()
                    df_results['Prediccion'] = predictions
                    st.dataframe(df_results, use_container_width=True)

                except Exception as e:
                    st.error(f"Ocurrió un error durante la ejecución: {e}")

# --- PESTAÑA 2: Saludo ---
with tab2:
    st.container()
    st.header("Respuesta a campañas publicitarias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Sube aquí las variables predictoras (X_test)")
        x_test_file = st.file_uploader("Cargar X_test (CSV)", type=['csv'], key='333')

    with col2:
        st.info("Sube aquí los valores reales del Target (y_test)")
        y_test_file = st.file_uploader("Cargar y_test (CSV)", type=['csv'], key='444')

    st.markdown("---")

    # Botón de ejecución
    if st.button("⚡ Ejecutar Modelo y Predecir", type="primary", key='555'):
        # Validaciones previas
        if (x_test_file is None) or (y_test_file is None):
            st.warning("⚠️ Por favor, sube los archivos X_test e y_test antes de continuar.")
        else:
            # 1. Cargar el modelo
            model = cargar_modelo(MODEL_PATH_Target_Campaign_Responsiveness)
            
            if model is None:
                st.error(f"❌ No se encontró el archivo '{MODEL_PATH_Target_Campaign_Responsiveness}' en el directorio.")
                st.markdown("**Consejo:** Asegúrate de que tu archivo .pkl esté en la misma carpeta o actualiza la variable `MODEL_PATH`.")
            else:
                try:
                    # 2. Leer los CSV
                    df_x = pd.read_csv(x_test_file, index_col=0)
                    df_y = pd.read_csv(y_test_file, index_col=0)

                    y_test = df_y["Target_Campaign_Responsiveness"]
                    
                    # 3. Realizar predicción
                    st.success("✅ Modelo cargado y datos leídos correctamente. Generando predicciones...")
                    predictions = model.predict(df_x)
                    y_proba = model.predict_proba(df_x)[:, 1]

                    acc = accuracy_score(y_test, predictions)
                    roc = roc_auc_score(y_test, y_proba)
                    rec = recall_score(y_test, predictions)      # Importante: Capacidad de encontrar los 1s
                    prec = precision_score(y_test, predictions)  # Importante: Cuántos de los que dijimos 1 son realmente 1
                    f1 = f1_score(y_test, predictions)

                    results = {
                        'ROC AUC': roc, 
                        'Recall (Sensibilidad)': rec, 
                        'Precision': prec,
                        'F1 Score': f1,
                        'Accuracy': acc
                    }
                    
                    # Métricas de la predicción
                    st.write("### Métricas de la predicción:")
                    for metric_name, metric_value in results.items():
                        st.write(f"##### {metric_name}:", np.round(metric_value,2))

                    # 4. Generar Gráfico de ejemplo (Matplotlib + Seaborn)
                    st.write("### Análisis Gráfico")

                    #####################################################################
                    # Mostramos un gráfico donde se visualice predicción vs dato real
                    cm = confusion_matrix(y_test, predictions)
                    class_names = ['0 (NO responde)', '1 (SI responde)']

                    fig, ax = plt.subplots(figsize=(10, 6))

                    # 2. Dibujar el heatmap pasando el objeto 'ax'
                    sns.heatmap(
                        cm, 
                        annot=True,          # Valores numéricos
                        fmt='d',             # Enteros
                        cmap='Blues',        # Color
                        cbar=False,          # Sin barra lateral
                        xticklabels=class_names, 
                        yticklabels=class_names,
                        ax=ax                # <--- IMPORTANTE: le decimos a Seaborn que use nuestro eje
                    )

                    # 3. Configuración de títulos usando el objeto 'ax'
                    ax.set_title('Matriz de Confusión\n ', fontsize=16)
                    ax.set_ylabel('Valores Reales\n ', fontsize=12)
                    ax.set_xlabel('\n Predicciones del Modelo', fontsize=12)                        

                    #####################################################################

                    # Renderizar el gráfico en Streamlit
                    st.pyplot(fig)

                    # Mostrar los resultados
                    st.write("### Resultados de la predicción:")
                    df_results = df_x.copy()
                    df_results['Prediccion'] = predictions
                    st.dataframe(df_results, use_container_width=True)

                except Exception as e:
                    st.error(f"Ocurrió un error durante la ejecución: {e}")
