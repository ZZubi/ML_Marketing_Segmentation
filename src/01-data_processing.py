import pandas as pd
import numpy as np

# Leer fichero original
df = pd.read_csv('../data/raw/marketing_campaign.csv', sep = '\t')

# -------------------------------------------------------------------------------------------------------------------

# eliminamos las columnas que no aparecen descritas en la explicación del dataset:
df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

# -------------------------------------------------------------------------------------------------------------------

# La columna "Income" tiene unos pocos elementos establecidos a NULL. 
# Son apenas 24 filas de un total de 2240 (1.07%), por lo que procedo 
# a eliminarlas:
df = df.dropna(subset=['Income'])

# -------------------------------------------------------------------------------------------------------------------


# Tenemos 6 columnas similares que nos hablan de cuántas veces ha comprado el usuario 
# a través de una campaña, en relación a las últimas 6 campañas (AcceptedCmpX [X: 1 to 5] and Response columns).
# Vamos a unificar esta información en una única columna llamada "Target_Campaign_Responsiveness",
# que sencillamente sea un 0 si nunca ha aceptado una campaña, y 1 si ha aceptado alguna de las 6 últimas:

def get_campaign_responsiveness(row) -> int:
    if (row['AcceptedCmp1'] == 1) or (row['AcceptedCmp2'] == 1) or (row['AcceptedCmp3'] == 1) or (row['AcceptedCmp4'] == 1) or (row['AcceptedCmp5'] == 1) or (row['Response'] == 1):
        return 1

    return 0 # Otherwise return 0

df["Target_Campaign_Responsiveness"] = df.apply(lambda row: get_campaign_responsiveness(row), axis='columns') 

# Eliminamos las columnas originales:
df = df.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'])

# -------------------------------------------------------------------------------------------------------------------

# Vamos a transformar la columna Dt_Customer (fecha desde la que es cliente) 
# a un valor numérico que los modelos puedan entender (Dt_Customer_Unix_time_seconds):


df['tmp_datetime_column'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Convert the datetime objects to Unix time (in seconds) ---
# We use .astype(np.int64) // 10**9 to get the integer number of seconds.
# pd.to_datetime converts to nanoseconds (np.int64), so we divide by 10**9.
df['Dt_Customer_Unix_time_seconds'] = df['tmp_datetime_column'].astype(np.int64) // 10**9

# Eliminamos las columnas original y temporal:
df = df.drop(columns=['tmp_datetime_column', 'Dt_Customer'])

# -------------------------------------------------------------------------------------------------------------------

# Columna "Education": Dado que no son muchos posibles valores, y que queremos que sea lo más compatible 
# con todo tipo de modelos, lo convertimos con un One-Hot encoding:
df = pd.get_dummies(df, dtype=int, columns=["Education"])

# -------------------------------------------------------------------------------------------------------------------

# Columna "Marital_Status": Dado que no son muchos posibles valores, y que queremos que sea lo más compatible 
# con todo tipo de modelos, lo convertimos con un One-Hot encoding:
df = pd.get_dummies(df, dtype=int, columns=["Marital_Status"])

# -------------------------------------------------------------------------------------------------------------------

# Columna "Year_Birth": Teniendo en cuenta que el dataset es de 2015, eliminamos los 3 clientes que tendrían más de 90 años, pues no queremos enviar la campaña a estos clientes 
# que posiblemente hayan fallecido o no estén en edad de comprar

df = df[df["Year_Birth"] >= 1925]

# -------------------------------------------------------------------------------------------------------------------

# Columna "Income": Observamos que el valor de 666666 es único y muy alejado del resto de outliers. 
# Decidimos eliminar este registro y mantener el resto de outliers que parecen más realistas:
df = df[df["Income"] < 666666]

# -------------------------------------------------------------------------------------------------------------------

# Vamos a transformar las columnas de gasto en diferentes artículos para que reflejen la media de gasto anual en lugar del gasto total de los últimos dos años
# Aprovechamos a marcar la columna relativa a "MntWines" con el prefijo Target
df["Target_MntWinesYear"] = df["MntWines"] / 2
df["MntFruitsYear"] = df["MntFruits"] / 2
df["MntMeatProductsYear"] = df["MntMeatProducts"] / 2
df["MntFishProductsYear"] = df["MntFishProducts"] / 2
df["MntSweetProductsYear"] = df["MntSweetProducts"] / 2
df["MntGoldProdsYear"] = df["MntGoldProds"] / 2

# Borramos las columnas originales
df = df.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'])

# -------------------------------------------------------------------------------------------------------------------

# Guardamos el fichero resultante en un nuevo CSV:
df.to_csv('../data/processed/clean_data.csv', index=False)