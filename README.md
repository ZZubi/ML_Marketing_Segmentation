# Proyecto de Segmentación y Predicción de Valor de Clientes

Este repositorio contiene el flujo de trabajo completo para el análisis, procesamiento y modelado de datos de una campaña de marketing. El objetivo es predecir el comportamiento de gasto de los clientes, su respuesta a campañas y segmentarlos para estrategias personalizadas.

## Estructura de Trabajo

El proyecto se estructura en 4 notebooks secuenciales:

### 1. Análisis, Limpieza y Transformación (01-...)

* **Objetivo:** Preparar el "raw data" para el modelado.

* **Acciones:**

    * Carga de datos (marketing_campaign.csv).

    * Limpieza de valores nulos y eliminación de outliers.

    * Feature Engineering: Creación de variables agregadas anuales (ej. MntWinesYear) y binarización de la respuesta a campañas (Target_Campaign_Responsiveness).

    * Output: Generación del dataset limpio clean_data.csv.

### 2. Modelos de Regresión: Gasto en Vino (02-...)

* **Objetivo:** Predecir la cantidad anual que un cliente gastará en vino (Target_MntWinesYear).

* **Modelos evaluados:** XGBoost Regressor, Random Forest y SVM.

* **Proceso:** Optimización de hiperparámetros con GridSearchCV y validación cruzada.

* **Conclusión:** XGBoost ofreció el mejor rendimiento (menor RMSE), identificando las variables más influyentes en el gasto.

### 3. Modelos de Clasificación: Respuesta a Campañas (03-...)
* **Objetivo:** Predecir la probabilidad de que un cliente acepte una oferta en la próxima campaña (Target_Campaign_Responsiveness).

* **Modelos evaluados:** XGBoost Classifier, Random Forest y SVM.

* **Proceso:** Manejo de clases desbalanceadas y optimización buscando maximizar el Recall (captar a la mayoría de clientes positivos).

* **Conclusión:** Random Forest obtuvo el mejor balance en métricas de sensibilidad (Recall) y precisión global.

### 4. Modelos No Supervisados: Segmentación (04-...)
* **Objetivo:** Agrupar clientes en perfiles homogéneos sin etiquetas previas.

* **Modelo:** K-Means Clustering.

* **Resultados:**

    * Determinación del número óptimo de clusters (K=4) mediante el método del codo y Silhouette.

    * Visualización 3D de los segmentos.

    * Perfilado: Identificación de grupos de "Alto Valor" (alto gasto y alta respuesta) frente a grupos de bajo engagement.

---

**Stack Tecnológico:** Python, Pandas, Scikit-Learn, XGBoost, Seaborn, Matplotlib.

---

## CSV file Attributes

Input data obtained from [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

### Predictive variables

* Customer

    * ID: Customer's unique identifier
    * Year_Birth: Customer's birth year
    * Education: Customer's education level
    * Marital_Status: Customer's marital status
    * Income: Customer's yearly household income
    * Kidhome: Number of children in customer's household
    * Teenhome: Number of teenagers in customer's household
    * Dt_Customer: Date of customer's enrollment with the company
    * Recency: Number of days since customer's last purchase
    * Complain: 1 if the customer complained in the last 2 years, 0 otherwise

* Place

    * NumWebPurchases: Number of purchases made through the company’s website
    * NumCatalogPurchases: Number of purchases made using a catalogue
    * NumStorePurchases: Number of purchases made directly in stores
    * NumWebVisitsMonth: Number of visits to company’s website in the last month

* Products

    * MntFruits: Amount spent on fruits in last 2 years
    * MntMeatProducts: Amount spent on meat in last 2 years
    * MntFishProducts: Amount spent on fish in last 2 years
    * MntSweetProducts: Amount spent on sweets in last 2 years
    * MntGoldProds: Amount spent on gold in last 2 years

### Target variables:

* __[Regression]__ Target_MntWinesYear; based on input field:
    * MntWines: Amount spent on wine in last 2 years

* __[Classification]__ Target_Campaign_Responsiveness; based on input fields:

    * AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
    * AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
    * AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
    * AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
    * AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
    * Response: 1 if customer accepted the offer in the last campaign, 0 otherwise



