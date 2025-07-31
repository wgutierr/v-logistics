#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import MSTL
from math import sqrt
from scipy.stats import norm


# # Funciones de ayuda pronosticos

# In[2]:


def crear_columnas_error(df):
    
    df['ERROR'] = df['DEMANDA'] - df['FORECAST'] # Error
    df['ABS_ERROR'] = df['ERROR'].abs() # Error Absoluto
    df['ERROR_PORC'] = np.where(df['DEMANDA'] == 0, 2, df['ABS_ERROR'] / df['DEMANDA']) # Error porcentual, devuelve 200% si la demanda es 0
    df['ERROR_CUADRADO'] = df['ERROR'] ** 2 # Error al cuadrado
    
    return df


# In[3]:


def calcular_error(df):
    df['MAE%'] = df['ABS_ERROR']/df['DEMANDA']
    df['SESGO%'] = df['ERROR']/df['DEMANDA']
    df['SCORE%'] = df['MAE%'] + df['SESGO%'].abs()
    if 'ERROR_CUADRADO_suma' in df.columns:
        df['RMSE'] = np.sqrt(df['ERROR_CUADRADO_suma'] / df['ERROR_CUADRADO_cuenta'])
    return df


# In[4]:


def metricas_error(df, imprimir):
     
    # Verificar si el total de la demanda es 0
    if df['DEMANDA'].sum() == 0:
        sesgo_porc = 2
        mae_porc = 2
        score = 2
    else:
        sesgo_porc = df['ERROR'].sum() / df['DEMANDA'].sum()
        mae_porc = df['ABS_ERROR'].sum() / df['DEMANDA'].sum()
        score = mae_porc + abs(sesgo_porc)
    
    rmse = np.sqrt(df['ERROR_CUADRADO'].mean())
        # Muestra los resultados formateados
    if imprimir == 1:
        print('MAE% modelo: {:.2%}'.format(mae_porc))
        print('Sesgo% modelo: {:.2%}'.format(sesgo_porc))
        print('Score modelo: {:.2%}'.format(score))
        print('RMSE modelo: {:.1f}'.format(rmse))
   
    return sesgo_porc, mae_porc, rmse, score


# In[5]:


def kpi_error_sku(df):
    
    if df is None:
        return None, None, None
        
    # Definicion de fechas de testeo
    if 'CONSECUTIVO' in df.columns:
        fecha_fin_testeo = df['CONSECUTIVO'].max()
        fecha_inicio_testeo = df['CONSECUTIVO'].min()
    else:    
        fecha_fin_testeo = df['Turn'].max()
        fecha_inicio_testeo = df['Turn'].min()

    # Crear columnas de error para cada pronostico generado
    df_test = crear_columnas_error(df)

    # Imprimir informacion de los periodos evaluados
    print('Periodo de Evaluacion desde:')
    if 'CONSECUTIVO' in df.columns:
        print(f"\033[1m{df_test['CONSECUTIVO'].min()} hasta {df_test['CONSECUTIVO'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    else:
        print(f"\033[1m{df_test['Turn'].min()} hasta {df_test['Turn'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    
    # Calcular metricas de error
    sesgo_porc, mae_porc, rmse, score = metricas_error(df_test, imprimir=1)
    
    # Agrupar df por sku
    grupo_sku_error = df_test.groupby(['PRODUCTO','REGIONAL'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_error.columns = ['PRODUCTO', 'REGIONAL','DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por sku
    grupo_sku_error = calcular_error(grupo_sku_error)
    
    # Ordenar el DataFrame por 'SCORE%' en orden ascendente
    grupo_sku_error = grupo_sku_error.sort_values(by='SCORE%')
    
    # Aplicar formato porcentaje
    formatted_columns = grupo_sku_error[['MAE%', 'SESGO%', 'SCORE%']].map(lambda x: f'{x * 100:.2f}%')
    
    # Concatenar la columna "PRODUCTO" sin formatear con las columnas formateadas
    grupo_sku_error_formato = pd.concat([grupo_sku_error[['PRODUCTO','REGIONAL']], formatted_columns], axis=1)
    
    # Mostrar el resultado
    display(grupo_sku_error_formato)

    # Agrupar por PRODUCTO y por Lag para almacenar RMSE
    grupo_sku_lag_error = df_test.groupby(['PRODUCTO', 'REGIONAL','LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_lag_error.columns = ['PRODUCTO','REGIONAL','LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por lag
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)

    # Calcular error rmse por lag
    rmse_sku_lag = grupo_sku_lag_error[['PRODUCTO','REGIONAL','LAG','RMSE']]
    
    # Agrupar por PRODUCTO para almacenar RMSE
    #df_test['Mes'] = df_test.index.month
    grupo_sku_mes_error = df_test.groupby(['PRODUCTO', 
                                           'REGIONAL'
                                          ], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()
    
    # Renombrar columnas
    grupo_sku_mes_error.columns = ['PRODUCTO',
                                  'REGIONAL', 
                                   'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']

    # Calcular error rmse por PRODUCTO
    grupo_sku_mes_error = calcular_error(grupo_sku_mes_error)

    # Filtrar las columnas para mejor visualizacion
    rmse_sku_mes = grupo_sku_mes_error[['PRODUCTO',
                                        'REGIONAL',
                                        'RMSE']]
    
    return grupo_sku_error_formato, rmse_sku_lag, rmse_sku_mes


# In[6]:


def kpi_error_lag(df):
    
    if df is None:
        return None, None
    # Definicion de fechas de testeo
    if 'CONSECUTIVO' in df.columns:
        fecha_fin_testeo = df['CONSECUTIVO'].max()
        fecha_inicio_testeo = df['CONSECUTIVO'].min()
    else:    
        fecha_fin_testeo = df['Turn'].max()
        fecha_inicio_testeo = df['Turn'].min()
    
    # Crear columnas de error  
    df_test = crear_columnas_error(df)
    print('Periodo de Evaluacion desde:')   
    if 'CONSECUTIVO' in df.columns:
        print(f"\033[1m{df_test['CONSECUTIVO'].min()} hasta {df_test['CONSECUTIVO'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla
    else:
        print(f"\033[1m{df_test['Turn'].min()} hasta {df_test['Turn'].max()}\033[0m") #\033[1m{}\033[0m muestra la linea en negrilla

    # Calcular loas metricas de error
    sesgo_porc, mae_porc, rmse, score = metricas_error(df_test, imprimir=1)
    
    # Agrupar df por mes
    grupo_mes_error = df_test.groupby(['LAG']).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_mes_error.columns = ['LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por mes
    grupo_mes_error = calcular_error(grupo_mes_error)
    
    # Aplicar formato porcentaje
    formatted_columns = grupo_mes_error[['MAE%', 'SESGO%', 'SCORE%']].map(lambda x: f'{x * 100:.2f}%')
    
    # Concatenar la columna "Lag" sin formatear con las columnas formateadas
    grupo_mes_error_formato = pd.concat([grupo_mes_error[['LAG']], formatted_columns], axis=1)
    
    # Mostrar el resultado
    display(grupo_mes_error_formato)

    # Agrupar por PRODUCTO y por Lag para almacenar RMSE
    grupo_sku_lag_error = df_test.groupby(['PRODUCTO','REGIONAL', 'LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()

    # Renombrar columnas
    grupo_sku_lag_error.columns = ['PRODUCTO','REGIONAL', 'LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']

    # Calcular columnas de error por lag
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)

    # Filtrar columnas para mejor visualizacion
    rmse_sku_lag = grupo_sku_lag_error[['PRODUCTO', 'REGIONAL','LAG','RMSE']]
    
    return grupo_mes_error_formato, df_test


# In[7]:


def agrupar_por_producto(df):
    grupo_sku_error = df.groupby(['PRODUCTO','REGIONAL'], observed=True).agg({
                                                                'DEMANDA': 'sum',
                                                                'ERROR': 'sum',
                                                                'ABS_ERROR': 'sum',
                                                                'ERROR_CUADRADO': ['sum', 'count'],
                                                                }).reset_index()
    grupo_sku_error.columns = ['PRODUCTO','REGIONAL', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                                 'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    # Calcular MAE% y Sesgo% de datos agregados por sku
    grupo_sku_error = calcular_error(grupo_sku_error)
    grupo_sku_error = grupo_sku_error[['PRODUCTO','REGIONAL','MAE%',	'SESGO%',	'SCORE%',	'RMSE']]
    
    # Agrupar por PRODUCTO y por Lag 
    grupo_sku_lag_error = df.groupby(['PRODUCTO','REGIONAL', 'LAG'], observed=True).agg({
                                                            'DEMANDA': 'sum',
                                                            'ERROR': 'sum',
                                                            'ABS_ERROR': 'sum',
                                                            'ERROR_CUADRADO': ['sum', 'count'],
                                                            }).reset_index()
    
    grupo_sku_lag_error.columns = ['PRODUCTO','REGIONAL','LAG', 'DEMANDA', 'ERROR', 'ABS_ERROR', 
                             'ERROR_CUADRADO_suma', 'ERROR_CUADRADO_cuenta']
    
    grupo_sku_lag_error = calcular_error(grupo_sku_lag_error)
    grupo_sku_lag_error = grupo_sku_lag_error[['PRODUCTO','REGIONAL','LAG','MAE%',	'SESGO%',	'SCORE%',	'RMSE']]

    # Pivotear el DataFrame de lag
    pivoted_lags = grupo_sku_lag_error.pivot(index=['PRODUCTO','REGIONAL'], columns='LAG', values='SCORE%')
    pivoted_lags.columns = [f"score_{col}" for col in pivoted_lags.columns]
    
    # Unir con el DataFrame principal
    tabla_final = grupo_sku_error.merge(pivoted_lags, on=['PRODUCTO','REGIONAL'], how='left')
    
    # Renombrar columnas para cumplir con el formato
    tabla_final = tabla_final.rename(columns={'MAE%': 'mae_porc', 'SESGO%': 'sesgo_porc', 'SCORE%': 'score', 'RMSE': 'rmse'})
    
    return tabla_final


# In[8]:


def adicionar_nombre_modelo_serie_tiempo(df, nombre_modelo):
    if df is None:
        return None
    df['MODELO'] = nombre_modelo
    df = df[['PRODUCTO','REGIONAL','Turn','FORECAST','LAG','MODELO']]
    
    return df


# In[9]:


def generar_reporte_error_skus(modelos):
    return {modelo: globals()[f'grupo_sku_error_formato_{modelo}'] for modelo in modelos}


# In[10]:


# Para pronosticos por meses
def comparar_y_graficar_modelos(reporte_error_skus):
    # Crear el DataFrame base con la columna 'PRODUCTO'
    df_final = reporte_error_skus['pms'][['PRODUCTO','REGIONAL']].copy()
    
    # Iterar sobre los modelos para combinarlos en df_final
    for nombre_modelo, df in reporte_error_skus.items():
        df_final = df_final.merge(
            #df[['PRODUCTO', 'MAE%']].rename(columns={'MAE%': nombre_modelo}), 
            df[['PRODUCTO','REGIONAL', 'SCORE%']].rename(columns={'SCORE%': nombre_modelo}),
            on=['PRODUCTO', 'REGIONAL'],
            how='left'
        )
        df['MODELO'] = nombre_modelo
        
    # Remover simbolos de porcentaje y convertir columnas a valores numericos
    modelos_cols = list(reporte_error_skus.keys())
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda col: abs(col.str.rstrip('%').astype(float)))
    
    # Identificar la columna con el valor minimo para cada fila
    df_final['MEJOR_MODELO'] = df_final[modelos_cols].idxmin(axis=1)
    #dejar una copia sin formato porcentaje
    df_minimos = df_final.copy()
    # Dar formato a las columnas con un decimal y agregar el simbolo %
    df_final[modelos_cols] = df_final[modelos_cols].apply(lambda x: x.map('{:.1f}%'.format))
    
    # Contar cuantas veces el modelo es el mejor
    report = df_final['MEJOR_MODELO'].value_counts()
    
    # Preparar y crear la grafica de dona
    fig1 = go.Figure(data=[go.Pie(
        labels=report.index, 
        values=report.values, 
        hole=0.4,  
        textinfo='percent+label',  
        marker=dict(colors=px.colors.qualitative.Plotly)  
    )])
    
    # Actualizar Layout de la grafica
    fig1.update_layout(
        title='Distribucion de Mejor Modelo por SKUs',
        title_x=0.5,  
        template='plotly_white'  
    )
   


    # Concatenar todos los DataFrames en uno solo
    df_errores_totales = pd.concat(reporte_error_skus.values(), ignore_index=True) 
    
    return df_minimos, df_final, reporte_error_skus, fig1, df_errores_totales


# In[11]:


def concatenar_forecasts_pronosticos(modelos):
    # Filtrar los DataFrames válidos (no None y no vacíos)
    dfs_validos = [
        globals()[f'df_forecast_final_{modelo}']
        for modelo in modelos
        if globals()[f'df_forecast_final_{modelo}'] is not None and not globals()[f'df_forecast_final_{modelo}'].empty
    ]
    
    # Verificar si hay DataFrames válidos
    if not dfs_validos:
        print("No hay pronósticos válidos para concatenar.")
        return None

    # Concatenar todos los DataFrames válidos en uno solo
    df_todos_pronosticos = pd.concat(dfs_validos)

    # Asegurar que la columna 'PRODUCTO' sea de tipo string
    df_todos_pronosticos['PRODUCTO'] = df_todos_pronosticos['PRODUCTO'].astype(str)

    return df_todos_pronosticos


# In[12]:


def concatenar_rmse(modelos):
    # Obtener los DataFrames dinámicamente usando la lista de modelos
    dfs_error = []
    
    for modelo in modelos:
        # Obtener el DataFrame para cada modelo
        df = globals().get(f'rmse_sku_mes_{modelo}')
        
        # Verificar si el DataFrame es None o está vacío
        if df is None or df.empty:
            print(f"El modelo {modelo} fue ignorado porque no tiene datos.")
            continue
        
        # Añadir una columna 'MODELO' con el nombre del modelo
        df['MODELO'] = modelo
        #f['RMSE'] = np.ceil(df['RMSE']).astype(int)
        #df['RMSE'] = df['RMSE'].astype(int)
        # Añadir el DataFrame a la lista
        dfs_error.append(df)
    
    # Verificar si hay DataFrames para concatenar
    if not dfs_error:
        print("No hay datos para concatenar.")
        return pd.DataFrame()  # Devuelve un DataFrame vacío
    
    # Concatenar todos los DataFrames en uno solo
    df_todos_rmse = pd.concat(dfs_error, ignore_index=True)
    
    # Asegurar que la columna 'CODIGO' sea de tipo string
    df_todos_rmse['PRODUCTO'] = df_todos_rmse['PRODUCTO'].astype(str)

    return df_todos_rmse


# In[13]:


def obtener_mejor_pronostico(df_minimos, df_todos_pronosticos, df_errores_totales, df_todos_rmse):
    # Crear una lista para almacenar los DataFrames filtrados
    lista_filtrados = [
        df_todos_pronosticos[
            (df_todos_pronosticos['PRODUCTO'] == row['PRODUCTO']) &
            (df_todos_pronosticos['REGIONAL'] == row['REGIONAL']) &
            (df_todos_pronosticos['MODELO'] == row['MEJOR_MODELO'])
        ]
        for _, row in df_minimos.iterrows()
    ]
    
    # Concatenar todos los DataFrames filtrados
    df_pronosticos_mejor_modelo = pd.concat(lista_filtrados)
    
    # Pivotear el resultado para mostrar el forecast por Código, Modelo y Fecha
    #df_pronosticos_n_periodos = df_pronosticos_mejor_modelo.pivot_table(index=["PRODUCTO", "MODELO"], columns="FECHA", values="FORECAST")#.reset_index()
    df_pronosticos_finales = df_pronosticos_mejor_modelo.pivot_table(index=["PRODUCTO",'REGIONAL', "MODELO"], columns="Turn", values="FORECAST").reset_index()
    # Realizamos un merge para agregar las columnas coincidiendo por PRODUCTO y MODELO
    # Realiza el merge entre ambos DataFrames en las claves 'PRODUCTO' y 'MODELO'
    df_merged = pd.merge(
        df_pronosticos_finales, 
        df_errores_totales[['PRODUCTO','REGIONAL', 'MODELO', 'MAE%', 'SESGO%', 'SCORE%']], 
        on=['PRODUCTO','REGIONAL', 'MODELO'], 
        how='left'
    )
    df_merged_rmse = pd.merge(
        df_merged, 
        df_todos_rmse[['PRODUCTO', 'REGIONAL', 'MODELO', 'RMSE']], 
        on=['PRODUCTO','REGIONAL', 'MODELO'], 
        how='left'
    )
 
    # Inserta las columnas en las posiciones deseadas
    df_merged_rmse.insert(0, 'MAE%', df_merged_rmse.pop('MAE%'))
    df_merged_rmse.insert(1, 'SESGO%', df_merged_rmse.pop('SESGO%'))
    df_merged_rmse.insert(2, 'SCORE%', df_merged_rmse.pop('SCORE%'))
    df_merged_rmse.insert(3, 'RMSE', df_merged_rmse.pop('RMSE'))
    # Restaurar el índice anterior
    df_pronosticos_n_periodos = df_merged_rmse.set_index(['PRODUCTO', 'REGIONAL', 'MODELO'])
        
    return df_pronosticos_mejor_modelo, df_pronosticos_n_periodos


# In[14]:


def crear_grafica_pronostico(df, df_todos_pronosticos, df_pronosticos_mejor_modelo):
    
    # Crear una figura
    fig = go.Figure()
    
    # Lista para almacenar las trazas
    trazas = []
    trazas_visibilidad = []
    
    productos_unicos = df['PRODUCTO'].unique()
    regionales_unicas = df['REGIONAL'].unique()
    modelos_unicos = df_todos_pronosticos['MODELO'].unique()

    # Combinación predeterminada
    producto_default = productos_unicos[0]
    regional_default = regionales_unicas[0]

    # Generar una paleta de colores en seaborn
    dark_colors = sns.color_palette("muted", n_colors=len(modelos_unicos)).as_hex()
    
    # Crear un diccionario para asignar colores a cada modelo
    color_mapping = {modelo: dark_colors[i] for i, modelo in enumerate(modelos_unicos)}
    
    # Crear todas las trazas (una por cada combinación de Producto, Regional y Modelo)
    for producto in productos_unicos:
        for regional in regionales_unicas:
            # Filtrar datos para DEMANDA
            df_filtrado = df[
                (df["PRODUCTO"] == producto) 
                & (df["REGIONAL"] == regional)
            ]
    
            # Filtrar datos para FORECAST de modelos
            df_todos_pronosticos_filtrado = df_todos_pronosticos[
                (df_todos_pronosticos["PRODUCTO"] == producto) & 
                (df_todos_pronosticos["REGIONAL"] == regional)
            ]
    
            # Filtrar para el mejor modelo
            df_pronosticos_filtrado = df_pronosticos_mejor_modelo[
                (df_pronosticos_mejor_modelo["PRODUCTO"] == producto) & 
                (df_pronosticos_mejor_modelo["REGIONAL"] == regional)
            ]
    
            if df_filtrado.empty or df_todos_pronosticos_filtrado.empty or df_pronosticos_filtrado.empty:
                continue
    
            # Extraer el mejor modelo
            mejor_modelo = df_pronosticos_filtrado["MODELO"].values[0]
    
            # Agregar traza de DEMANDA
            trazas.append(go.Scatter(
                x=df_filtrado['Turn'], 
                y=df_filtrado["DEMANDA"], 
                mode='lines',
                name=f'Demanda {producto} - {regional}',
                line=dict(color='navy'),
                visible=(producto == producto_default and regional == regional_default)
            ))
            trazas_visibilidad.append((producto, regional, 'DEMANDA'))
    
            # Agregar trazas de FORECAST para todos los modelos
            for modelo in modelos_unicos:
                df_modelo_filtrado = df_todos_pronosticos_filtrado[
                    df_todos_pronosticos_filtrado["MODELO"] == modelo
                ]
                if df_modelo_filtrado.empty:
                    continue
    
                line_style = dict(
                    dash='solid' if modelo == mejor_modelo else 'dot',
                    color=color_mapping[modelo],
                    width=2.5 if modelo == mejor_modelo else 1.5
                )
    
                trazas.append(go.Scatter(
                    x=df_modelo_filtrado['Turn'], 
                    y=df_modelo_filtrado["FORECAST"], 
                    mode='lines',
                    name=f'{modelo}',
                    line=line_style,
                    visible=(producto == producto_default and regional == regional_default)
                ))
                trazas_visibilidad.append((producto, regional, modelo))
    
    # Agregar trazas a la figura
    for traza in trazas:
        fig.add_trace(traza)
    
    # Función para generar la visibilidad
    def generar_visibilidad(producto_seleccionado, regional_seleccionado):
        """Genera una lista de visibilidad para las trazas."""
        return [
            p == producto_seleccionado and r == regional_seleccionado
            for p, r, _ in trazas_visibilidad
        ]
    
    # Crear menús desplegables
    buttons = []
    for producto in productos_unicos:
        for regional in regionales_unicas:
            buttons.append(
                dict(
                    label=f"{producto} - {regional}",
                    method="update",
                    args=[
                        {"visible": generar_visibilidad(producto, regional)},
                        {"title": f"Producto: {producto} | Regional: {regional}"}
                    ]
                )
            )
    
    # Actualizar el layout con los menús
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                y=1.15,
                xanchor="left",
                yanchor="top"
            )
        ],
        title=f"Producto: {producto_default} | Regional: {regional_default}",
        xaxis_title="Turn",
        yaxis_title="Valores"
    )
    return fig


# # Holt Winters

# In[15]:


def aplicar_hw(df, productos, regionales, lags, pronostico_final=0):
    resultados_pronostico_sku = []

    for producto in productos:
        for regional in regionales:
            # Filtrar por producto y regional
            df_producto_regional = df[(df['REGIONAL'] == regional) & (df['PRODUCTO'] == producto)]
            if df_producto_regional.empty:
                continue

            serie = df_producto_regional['DEMANDA']
            serie.index = df_producto_regional['Turn']  # Índice basado en "Turn"
            
            # Verificar valores NaN
            if serie.isna().any():
                print(f"Advertencia: La serie para {producto}, {regional} contiene valores NaN. No se pronosticará.")
                continue
                
            max_rango = df['Turn'].max()
            min_rango = -35 if pronostico_final == 0 else max_rango
                        
            for i in range(min_rango, max_rango+1):
                # Dividir en entrenamiento (reseteando el índice)
                train_index = serie.loc[:i]
                train = train_index.reset_index(drop=True)
    
                # Ajustar el modelo Holt-Winters
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    modelo = ExponentialSmoothing(
                        train,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=52
                    ).fit()
                    
                # Generar pronóstico
                pronostico = modelo.forecast(lags)
                
                # Guardar resultados
                for lag, valor in enumerate(pronostico, start=1):
                    resultados_pronostico_sku.append({
                        'PRODUCTO': producto,
                        'REGIONAL': regional,
                        'Turn': train_index.index.max() + lag,  # Índice numérico
                        'LAG': lag,
                        'FORECAST': valor
                    })

    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados_pronostico_sku)

    if pronostico_final == 0:
        df_forecast_hw = df_resultados.merge(df, on=['Turn', 'REGIONAL', 'PRODUCTO'], how='left').dropna()

    else:
        df_forecast_hw =  df_resultados
    df_forecast_hw['FORECAST'] = df_forecast_hw['FORECAST'].round()    
    #df_forecast_hw['FORECAST'] = np.ceil(df_forecast_hw['FORECAST'])    
    #df_forecast_hw['FORECAST'] = np.floor(df_forecast_hw['FORECAST'])
    return df_forecast_hw
    


# # Suavizacion Exponencial

# In[16]:


def aplicar_se(df, productos, regionales, lags, pronostico_final=0):
    resultados_pronostico_sku = []

    for producto in productos:
        for regional in regionales:
            # Filtrar por producto y regional
            df_producto_regional = df[(df['REGIONAL'] == regional) & (df['PRODUCTO'] == producto)]
            if df_producto_regional.empty:
                continue

            serie = df_producto_regional['DEMANDA']
            serie.index = df_producto_regional['Turn']  # Índice basado en "Turn"
            
            # Verificar valores NaN
            if serie.isna().any():
                print(f"Advertencia: La serie para {producto}, {regional} contiene valores NaN. No se pronosticará.")
                continue
                
            max_rango = df['Turn'].max()
            min_rango = -35 if pronostico_final == 0 else max_rango
                        
            for i in range(min_rango, max_rango+1):
                # Dividir en entrenamiento (reseteando el índice)
                train_index = serie.loc[:i]
                train = train_index.reset_index(drop=True)
                
                # Ajustar el modelo Holt-Winters
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    modelo = SimpleExpSmoothing(
                        train
                    ).fit(smoothing_level=None, optimized=True)
                                    
                # Generar pronóstico
                pronostico = modelo.forecast(lags)
                
                # Guardar resultados
                for lag, valor in enumerate(pronostico, start=1):
                    resultados_pronostico_sku.append({
                        'PRODUCTO': producto,
                        'REGIONAL': regional,
                        'Turn': train_index.index.max() + lag,  # Índice numérico
                        'LAG': lag,
                        'FORECAST': valor
                    })

    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados_pronostico_sku)
    
    if pronostico_final == 0:
        df_forecast_se = df_resultados.merge(df, on=['Turn', 'REGIONAL', 'PRODUCTO'], how='left').dropna()

    else:
        df_forecast_se =  df_resultados
    df_forecast_se['FORECAST'] = df_forecast_se['FORECAST'].round() 
    return df_forecast_se


# # Promedio movil simple

# In[17]:


def aplicar_pms_mp(df, productos, regionales, lags, pronostico_final=0):
    pronostico_n = []  
    for producto in productos:
            for regional in regionales:
                # Filtrar por producto y regional
                df_producto_regional = df[(df['REGIONAL'] == regional) & (df['PRODUCTO'] == producto)]
                if df_producto_regional.empty:
                    continue
    
                serie = df_producto_regional['DEMANDA']
                serie.index = df_producto_regional['Turn']  # Índice basado en "Turn"
                
                # Verificar valores NaN
                if serie.isna().any():
                    print(f"Advertencia: La serie para {producto}, {regional} contiene valores NaN. No se pronosticará.")
                    continue

                max_rango = df['Turn'].max()
                min_rango = -35 if pronostico_final == 0 else max_rango
                            
                for i in range(min_rango, max_rango+1):
                
                #for i in range(-35, 0):
                    score_n = []
                    for n in range(3, 15):        
                        # Dividir en entrenamiento (reseteando el índice)
                        train = serie.loc[:i]
                        # Ajustar el modelo PMS
                        serie_rolling_mean_shifted = train.rolling(window=n, min_periods=1).mean().shift(1)
                        # Crear la serie 'error' restando las dos series
                        error = (train - serie_rolling_mean_shifted).dropna()                    
                        # Crear la serie 'error_abs' con los valores absolutos
                        error_abs = error.abs()                    
                        # Calcular la variable mae_porc                                       
                        mae_porc = error_abs.sum() / train.loc[error.index].sum()                    
                        # Calcular la variable sesgo
                        sesgo = error.sum() / train.loc[error.index].sum()                    
                        # Calcular Score
                        score = mae_porc + abs(sesgo)
                        #print('n:',n,'Score:',score)
                        score_n.append({
                            'n':n,
                            'score':score
                            })
                    #print('Turn:', i)
                    #print(score_n)
                    # Encontrar el diccionario con el menor score
                    mejor_score = min(score_n, key=lambda x: x['score'])
                    # Obtener el valor de 'n' con el menor score
                    mejor_n = mejor_score['n']
                    # Identificar menor score
                    score_minimo = mejor_score['score']
                    # Generar pronostico con mejor_n
                    pronostico = [train.rolling(window=mejor_n, min_periods=1).mean().iloc[-1]]*lags
                    
                    for lag, valor in enumerate(pronostico, start=1):
                        pronostico_n.append({
                            'Turn': train.index.max() + lag,
                            'PRODUCTO': producto,
                            'REGIONAL': regional,
                            'LAG': lag,
                            'FORECAST': valor,
                            'MEJOR_n': mejor_n
                        })
    df_resultados = pd.DataFrame(pronostico_n)
    if pronostico_final == 0:
        df_forecast_pms = df_resultados.merge(df, on=['Turn', 'REGIONAL', 'PRODUCTO'], how='left').dropna()

    else:
        df_forecast_pms =  df_resultados
    df_forecast_pms['FORECAST'] = df_forecast_pms['FORECAST'].round() 
    return df_forecast_pms

