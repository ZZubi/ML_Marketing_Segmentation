import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de la página
st.set_page_config(page_title="App de Predicción ML", layout="wide")

# Ruta donde se espera que esté el modelo
MODEL_PATH = 'modelo_entrenado.pkl'

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
st.title("🔮 Dashboard de Predicción de Machine Learning")

# Definimos las pestañas (Tabs)
tab1, tab2 = st.tabs(["🚀 Ejecución del Modelo", "👋 Saludo"])

# --- PESTAÑA 1: Ejecución del Modelo ---
with tab1:
    st.header("Carga de datos y Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Sube aquí tus variables predictoras")
        x_test_file = st.file_uploader("Cargar X_test (CSV)", type=['csv'])

    with col2:
        st.info("Sube aquí tus etiquetas reales (opcional para predicción, útil para validar)")
        y_test_file = st.file_uploader("Cargar y_test (CSV)", type=['csv'])

    st.markdown("---")

    # Botón de ejecución
    if st.button("⚡ Ejecutar Modelo y Predecir", type="primary"):
        # Validaciones previas
        if (x_test_file is None) or (y_test_file is None):
            st.warning("⚠️ Por favor, sube los archivos X_test e y_test antes de continuar.")
        else:
            # 1. Cargar el modelo
            model = cargar_modelo(MODEL_PATH)
            
            if model is None:
                st.error(f"❌ No se encontró el archivo '{MODEL_PATH}' en el directorio.")
                st.markdown("**Consejo:** Asegúrate de que tu archivo .pkl esté en la misma carpeta o actualiza la variable `MODEL_PATH`.")
            else:
                try:
                    # 2. Leer el CSV
                    df_x = pd.read_csv(x_test_file)
                    
                    # 3. Realizar predicción
                    st.success("✅ Modelo cargado y datos leídos correctamente. Generando predicciones...")
                    predictions = model.predict(df_x)
                    
                    # Mostrar un vistazo de los resultados
                    st.write("### Resultados de la predicción:")
                    df_results = df_x.copy()
                    df_results['Prediccion'] = predictions
                    st.dataframe(df_results.head())

                    # 4. Generar Gráfico de ejemplo (Matplotlib + Seaborn)
                    st.write("### Análisis Gráfico")
                    
                    # Creamos datos de ejemplo para el gráfico (o usamos los datos subidos si son numéricos)
                    # Aquí generamos un gráfico de ejemplo cualquiera como pediste.
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Ejemplo: Histograma de una distribución normal simulada
                    # (para asegurar que funcione independientemente de los datos del CSV)
                    import numpy as np
                    data_plot = np.random.randn(1000)
                    
                    sns.histplot(data_plot, kde=True, color="teal", ax=ax)
                    ax.set_title("Gráfico de Ejemplo: Distribución de Resultados")
                    ax.set_xlabel("Valor")
                    ax.set_ylabel("Frecuencia")
                    
                    # Renderizar el gráfico en Streamlit
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error durante la ejecución: {e}")

# --- PESTAÑA 2: Saludo ---
with tab2:
    st.container()
    st.write("## Hello World!")
    st.write("Esta es la segunda sección de la aplicación.")