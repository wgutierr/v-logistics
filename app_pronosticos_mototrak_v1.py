# %%
# Funciones basicas
import pandas as pd
import numpy as np

# Funciones de graficación
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Funciones de pronóstico y estadisticas
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
import math
from math import sqrt
from scipy.stats import norm

# Funciones de manejo de fechas
from datetime import datetime, timedelta

# Funciones de manejo de texto
import re

# Funciones de manejo de archivos
import os
import io

# Funciones de manejo de excepciones
import sys
# Funciones para la interfaz de usuario
try:
    import streamlit as st
    USANDO_STREAMLIT = 'streamlit' in sys.modules
except ImportError:
    st = None
    USANDO_STREAMLIT = False



# %% [markdown]
# # Funciones de Apoyo de Carga de Datos

# %% [markdown]
# ## Carga de datos de demanda

# %%
def cargar_demandas(ruta_demandas):

    # Lista para almacenar cada DataFrame
    dataframes = []

    # Itera sobre cada archivo en la carpeta
    for filename in os.listdir(ruta_demandas):
        if filename.endswith('2025.csv'):
            # Extrae "Producto" y "Regional" del nombre del archivo
            regional, año = filename.split('_')
            año = año.replace('.csv', '')
            
            # Carga el archivo y añade las columnas "Producto" y "Regional"
            df = pd.read_csv(os.path.join(ruta_demandas, filename))
            df['REGIONAL'] = regional
            print(f'Ultimo turno {regional}:',df['Turn'].max())
            # Agrega el DataFrame a la lista
            dataframes.append(df)

    # Concatena todos los DataFrames en uno solo
    df_agregado = pd.concat(dataframes, ignore_index=True)

    return df_agregado


# %%
def cargar_demandas_por_region(archivo_norte, archivo_centro, archivo_sur):
    
    """
    Carga y concatena los archivos de demanda por región (NORTE, CENTRO, SUR) 
    desde archivos subidos vía Streamlit.
    """

    dataframes = []

    archivos = {
        'NORTE': archivo_norte,
        'CENTRO': archivo_centro,
        'SUR': archivo_sur
    }

    for region, archivo in archivos.items():
        if archivo is not None:
            df = pd.read_csv(archivo)
            df['REGIONAL'] = region
            st.write(f"✅ Último turno cargado para {region}: {df['Turn'].max()}")
            dataframes.append(df)

    df_agregado = pd.concat(dataframes, ignore_index=True)

    return df_agregado

# %% [markdown]
# ## Carga de datos maestros

# %%
def cargar_data_maestra(ruta_data_maestra):

    # Carga todas las hojas como diccionario
    hojas = pd.read_excel(ruta_data_maestra, sheet_name=None)  

    # Crear un DataFrame por cada hoja con nombre df_{nombre_hoja}
    for nombre_hoja, df in hojas.items():
        # Limpiar y estandarizar el nombre de la hoja
        nombre_limpio = re.sub(r'\W+', '_', nombre_hoja.lower())  # Minúsculas y reemplazo de no alfanuméricos por "_"
        globals()[f"df_{nombre_limpio}"] = df

    # (Opcional) Verificar los nombres creados
    print("Hojas cargadas:", [f"df_{re.sub(r'\\W+', '_', nombre.lower())}" for nombre in hojas.keys()])
    
    return df_bom_mp, df_m_d_o, df_transporte, df_almacenamiento

# %% [markdown]
# # Funciones de Apoyo para Preprocesamiento de Datos

# %% [markdown]
# ### Producto terminado Parte 1

# %%
def preprocesar_datos_parte_1(df_agregado, productos):

    """
    Toma las demandas agregadas y la lista de productos,
    limpia los nombres de columnas, estandariza los nombres de las regionales,
    genera dos agregados adicionales:
    - 'CEDI': suma de CENTRO + SUR
    - 'MOTOTRAK': suma de NORTE + CENTRO + SUR
    Concatena estos agregados al DataFrame original y lo ordena.
    Devuelve el DataFrame final listo para análisis o modelado.
    """

    # Limpiar nombres de columnas
    df_agregado.columns = df_agregado.columns.str.replace(r"\s*\(Product\)", "", regex=True).str.strip()
    df_agregado['REGIONAL'] = df_agregado['REGIONAL'].str.upper()

    # Crear CEDI: CENTRO + SUR
    df_cedi = df_agregado[df_agregado['REGIONAL'].isin(['CENTRO', 'SUR'])].groupby('Turn')[productos].sum().reset_index()
    df_cedi['REGIONAL'] = 'CEDI'

    # Crear MOTOTRAK: NORTE + CENTRO + SUR
    df_mototrak = df_agregado[df_agregado['REGIONAL'].isin(['NORTE', 'CENTRO', 'SUR'])].groupby('Turn')[productos].sum().reset_index()
    df_mototrak['REGIONAL'] = 'MOTOTRAK'

    # Concatenar todo
    df_final = pd.concat([df_agregado, df_cedi, df_mototrak], ignore_index=True)

    # Mostar df
    return df_final

# %% [markdown]
# ### Producto terminado Parte 2

# %%
def preprocesar_datos_parte_2(df_final):
    
    """    Transforma el DataFrame df_final para que las columnas de productos
    ('MOTO', 'CUATRIMOTO', 'TRACTOR') se conviertan en filas,
    manteniendo 'Turn' y 'REGIONAL' como columnas fijas.
    """

    # Transformar el DataFrame utilizando pd.melt
    df = pd.melt(
        df_final, 
        id_vars=['Turn', 'REGIONAL'],  # Columnas que permanecen fijas
        value_vars=['MOTO', 'CUATRIMOTO', 'TRACTOR'],  # Columnas que se convertirán en filas
        var_name='PRODUCTO',  # Nombre para la nueva columna de productos
        value_name='DEMANDA'  # Nombre para la nueva columna de valores
    )

    # Eliminar filas con DEMANDA nula (Tractor en Sur)
    df = df.dropna(subset='DEMANDA').reset_index(drop=True)

    # Visualizar el resultado
    return df

# %% [markdown]
# ### Materia Prima Preprocesamiento BOM

# %%
def preprocesar_datos_mp(df_bom_mp):
    """
    Transforma el DataFrame df_bom_mp para que la columna de producto
    se conviertan en filas,
    manteniendo 'MATERIA_PRIMA' como columna fija.
    """
    # Seleccionar las columnas relevantes y renombrar 'PRODUCTO' a 'MATERIA_PRIMA'
    df_bom = df_bom_mp.rename(columns={'PRODUCTO':'MATERIA_PRIMA'}).iloc[:,:4]

    # Transformar el DataFrame utilizando pd.melt
    df_bom_vertical = df_bom.melt(id_vars=['MATERIA_PRIMA'], 
                                    var_name='PRODUCTO', 
                                    value_name='CANTIDAD')
    
    # Eliminar filas con CANTIDAD nula o cero
    df_bom_vertical = df_bom_vertical[df_bom_vertical['CANTIDAD'] != 0]
    
    return df_bom_vertical


# %% [markdown]
# ### Materia Prima - Explosión de Materiales

# %%
def explosionar_mp(df, df_bom_vertical):

    """
    Explosiona el DataFrame df_mototrak con los datos de la BOM vertical
    para calcular el consumo de cada material por Turno.
    """
    
    # Filtrar df_mototrak
    df_mototrak = df[df['REGIONAL'] == 'MOTOTRAK'].copy()

    # Paso 1: Unir df_mototrak con df_bom_vertical por 'PRODUCTO'
    df_explosion = df_mototrak.merge(df_bom_vertical, on='PRODUCTO', how='left')

    # Paso 2: Calcular el consumo de cada material por Turn
    df_explosion['CONSUMO'] = df_explosion['DEMANDA'] * df_explosion['CANTIDAD']

    # Paso 3: Agrupar por Turno y Materia Prima
    df_consumo = (
        df_explosion
        .groupby(['Turn', 'MATERIA_PRIMA'], as_index=False)
        .agg({'CONSUMO': 'sum'})
        .rename(columns={'CONSUMO': 'DEMANDA_MATERIA_PRIMA'})
    )

    # Mostrar el DataFrame resultante
    return df_consumo

# %% [markdown]
# # Funciones de Ayuda para Gráficas de Demanda

# %% [markdown]
# ## Función para graficar la demanda del producto terminado

# %%
def graficar_demanda_pt(df, colores_pt):

    """
    Crea un gráfico de líneas para la demanda de productos terminados
    por regionales y productos, utilizando Plotly.
    """

    # Listas de regionales y productos únicos
    regionales = df['REGIONAL'].unique().tolist()
    productos = df['PRODUCTO'].unique().tolist()

    # Crear figura 2x3
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=["NORTE", "CENTRO", "SUR", "CEDI (C+S)", "MOTOTRAK (N+C+S)", ""]
    )

    # Mapeo a subplot
    subplot_pos = {
        'NORTE': (1, 1),
        'CENTRO': (1, 2),
        'SUR': (1, 3),
        'CEDI': (2, 1),
        'MOTOTRAK': (2, 2)
    }

    # Mostrar leyenda solo en el primer subplot
    showlegend_flag = True

    # Trazar por cada regional
    for region in regionales:
        row, col = subplot_pos[region]
        df_region = df[df['REGIONAL'] == region]
        for producto in productos:
            df_sub = df_region[df_region['PRODUCTO'] == producto]
            if not df_sub.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_sub['Turn'], 
                        y=df_sub['DEMANDA'], 
                        mode='lines',
                        name=producto,
                        line=dict(color=colores_pt[producto]),
                        showlegend=showlegend_flag
                    ),
                    row=row, col=col
                )
        showlegend_flag = False  # Solo en el primer gráfico

    # Layout
    fig.update_layout(
        height=700, width=1200,
        title_text="Demanda por Regional y Agregados",
        showlegend=True,
        legend_title="Producto",
        template="ggplot2"
    )

    # Etiquetas comunes
    fig.update_xaxes(title_text="Turn", row=2, col=1)
    fig.update_xaxes(title_text="Turn", row=2, col=2)
    fig.update_yaxes(title_text="Demanda", row=1, col=1)
    fig.update_yaxes(title_text="Demanda", row=2, col=1)

    # Mostrar gráfico
    fig.show()

# %% [markdown]
# ## Función para graficar los pronósticos de productos terminados

# %%
def graficar_pronosticos_pt(df, resultados_pt, colores_pt):

    # Listas de regionales y productos únicos
    regionales = df['REGIONAL'].unique().tolist()
    productos = df['PRODUCTO'].unique().tolist()

    # Crear figura 2x3
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["NORTE", "CENTRO", "SUR", "CEDI (C+S)", "MOTOTRAK (N+C+S)", ""]
    )

    # Posiciones de subplots
    subplot_pos = {
        'NORTE': (1, 1),
        'CENTRO': (1, 2),
        'SUR': (1, 3),
        'CEDI': (2, 1),
        'MOTOTRAK': (2, 2)
    }

    # Mostrar leyenda solo una vez
    showlegend_flag = True

    # Agregar trazos de demanda real y pronóstico
    for region in regionales:
        row, col = subplot_pos[region]
        df_region = df[df['REGIONAL'] == region]

        for producto in productos:
            # 1. Demanda real
            df_sub = df_region[df_region['PRODUCTO'] == producto]
            clave = (region, producto)

            if not df_sub.empty:
                # Si hay pronóstico, ajustar la longitud del histórico
                if clave in resultados_pt:
                    pronostico_final = resultados_pt[clave]["pronostico_final"]
                    mejor_modelo = resultados_pt[clave]["mejor_modelo"]

                    lags = len(pronostico_final)  # Cantidad de pasos de pronóstico
                    df_sub = df_sub.tail(52 + lags)  # Cortar a los últimos 52 + lags

                    fig.add_trace(
                        go.Scatter(
                            x=df_sub['Turn'],
                            y=df_sub['DEMANDA'],
                            mode='lines',
                            name=producto,
                            line=dict(color=colores_pt[producto]),
                            showlegend=showlegend_flag
                        ),
                        row=row, col=col
                    )

                    if not pronostico_final.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pronostico_final.index,
                                y=pronostico_final[mejor_modelo],
                                mode='lines',
                                name=f"{producto} ({mejor_modelo})",
                                line=dict(dash='dot', color=colores_pt[producto]),
                                showlegend=showlegend_flag
                            ),
                            row=row, col=col
                        )
                else:
                    # Si no hay pronóstico, igual limitar a últimos 52 datos
                    df_sub = df_sub.tail(52)
                    fig.add_trace(
                        go.Scatter(
                            x=df_sub['Turn'],
                            y=df_sub['DEMANDA'],
                            mode='lines',
                            name=producto,
                            line=dict(color=colores_pt[producto]),
                            showlegend=showlegend_flag
                        ),
                        row=row, col=col
                    )

        showlegend_flag = False  # Solo mostrar en el primer subplot

    # Layout final
    fig.update_layout(
        height=700, width=1200,
        title_text="Demanda Real y Pronóstico por Regional y Producto",
        showlegend=True,
        legend_title="Producto / Modelo",
        template="ggplot2"
    )

    # Etiquetas ejes
    fig.update_xaxes(title_text="Turn", row=2, col=1)
    fig.update_xaxes(title_text="Turn", row=2, col=2)
    fig.update_yaxes(title_text="Demanda", row=1, col=1)
    fig.update_yaxes(title_text="Demanda", row=2, col=1)

    return fig

# %% [markdown]
# ## Función para graficar los pronosticos de materia prima

# %%
def generar_colores_mp(elementos):
    """
    Asigna colores únicos a cada elemento (producto o materia prima).
    Usa una paleta de colores de Plotly.
    
    Parámetro:
    - elementos: lista o conjunto de nombres
    
    Retorna:
    - diccionario {elemento: color}
    """
    elementos = sorted(list(set(elementos)))
    paleta = px.colors.qualitative.Set2  # Puedes cambiar por Set1, Set2, Plotly, etc.
    n_colores = len(paleta)

    colores_mp = {
        elemento: paleta[i % n_colores]
        for i, elemento in enumerate(elementos)
    }

    return colores_mp

# %%
def graficar_pronosticos_mp(df, resultados_por_serie, colores_mp, lags=12):
    """
    Grafica series de consumo real y pronóstico para materias primas.

    Parámetros:
    - df: DataFrame con columnas ['Turn', 'MATERIA_PRIMA', 'DEMANDA_MATERIA_PRIMA']
    - resultados_por_serie: dict con claves = materia prima y valores con 'pronostico_final' y 'mejor_modelo'
    - colores_mp: dict {materia_prima: color}
    - lags: número de pasos de pronóstico
    """

    elementos = sorted(df['MATERIA_PRIMA'].unique())
    n = len(elementos)
    cols = 3
    rows = math.ceil(n / cols)

    # Dividir nombres largos con salto de línea si exceden cierto número de caracteres
    def ajustar_titulo(texto, max_len=30):
        return "<br>".join(texto[i:i+max_len] for i in range(0, len(texto), max_len))

    titulos = elementos

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titulos
    )

    showlegend_flag = True

    for i, materia in enumerate(elementos):
        row = (i // cols) + 1
        col = (i % cols) + 1

        fila = df[df['MATERIA_PRIMA'] == materia].tail(52 + lags)
        color = colores_mp.get(materia, None)

        fig.add_trace(
            go.Scatter(
                x=fila['Turn'],
                y=fila['DEMANDA_MATERIA_PRIMA'],
                mode='lines',
                name=materia,
                line=dict(color=color),
                showlegend=showlegend_flag
            ),
            row=row, col=col
        )

        if materia in resultados_por_serie:
            pronostico_final = resultados_por_serie[materia]["pronostico_final"]
            mejor_modelo = resultados_por_serie[materia]["mejor_modelo"]

            fig.add_trace(
                go.Scatter(
                    x=pronostico_final.index,
                    y=pronostico_final[mejor_modelo],
                    mode='lines',
                    name=f"{materia} ({mejor_modelo})",
                    line=dict(dash='dot', color=color),
                    showlegend=showlegend_flag
                ),
                row=row, col=col
            )

        showlegend_flag = False

    # Disminuir tamaño de fuente de títulos individuales
    for anotacion in fig['layout']['annotations']:
        anotacion['font'] = dict(size=11)

    fig.update_layout(
        height=300 * rows, width=1200,
        title_text="Consumo y Pronóstico por Materia Prima",
        showlegend=False,
        legend_title="Materia Prima / Modelo",
        template="ggplot2",
        font=dict(size=12)  # Solo afecta ejes, leyenda, título general
    )
    fig.update_xaxes(title_text="Turn")
    fig.update_yaxes(title_text="Demanda")

    return fig

# %% [markdown]
# # Funciones de Ayuda para Selección de pronósticos de Producto terminado

# %% [markdown]
# ## Creación de diccionario con series de tiempo producto-regional

# %%
# Crear un diccionario con cada serie de tiempo de demanda por producto y regional
def crear_dicc_pt(df):

    """
    Crea un diccionario donde las claves son tuplas (REGIONAL, PRODUCTO)
    y los valores son Series de DEMANDA indexadas por Turn.
    """
    
    series_dict_pt = {
        (reg, prod): serie
        for reg in df['REGIONAL'].unique()
        for prod in df['PRODUCTO'].unique()
        if not (serie := df[(df['REGIONAL'] == reg) & (df['PRODUCTO'] == prod)]
                    .set_index('Turn')['DEMANDA']
                    .sort_index()).empty
    }

    return series_dict_pt

# %% [markdown]
# ## Creación de diccionario con series de tiempo materia prima en mototrak

# %%
# Crear un diccionario con cada serie de tiempo de demanda por materia prima
def crear_dicc_mp(df_consumo):

    """
    Crea un diccionario con df de materia prima explosionada,
    y los valores son Series de DEMANDA indexadas por Turn.
    """

    series_dict_mp = {
        materia: serie
        for materia in df_consumo['MATERIA_PRIMA'].unique()
        if not (serie := df_consumo[df_consumo['MATERIA_PRIMA'] == materia]
                        .set_index('Turn')['DEMANDA_MATERIA_PRIMA']
                        .sort_index()).empty
    }

    return series_dict_mp

# %% [markdown]
# ## Backtesting
# Se hará backtesting desde n periodos hacia atras y generando múltiples pronósticos hacia adelante (lags)

# %%
def crear_pronosticos_generico(series_dict, periodos_atras=48, lags=6):
    """
    Aplica modelos de pronóstico sobre un diccionario de series univariadas.
    Funciona tanto para productos terminados como materias primas.
    """

    turnos = next(iter(series_dict.values())).index.tolist()
    rango_turnos = turnos[-(periodos_atras + 1):]
    resultados_por_serie = {}

    # Widgets dinámicos solo si estás en Streamlit
    progreso = st.empty() if USANDO_STREAMLIT else None
    barra = st.progress(0) if USANDO_STREAMLIT else None
    total = len(series_dict)

    for i, (clave, serie) in enumerate(series_dict.items()):
        if USANDO_STREAMLIT:
            progreso.markdown(f"👨‍💻 Analizando `{clave}`...")
            barra.progress((i + 1) / total)
        else:
            print(f"👨‍💻 Analizando {clave}")

        resultados_hw, resultados_hw_13 = [], []
        resultados_pm_3, resultados_pm_6, resultados_pm_12 = [], [], []

        for j, fecha_corte in enumerate(rango_turnos):
            serie_corte = serie[serie.index <= fecha_corte].copy()
            indice_real = serie_corte.index.copy()
            serie_corte.index = pd.RangeIndex(start=0, stop=len(serie_corte))

            inicio_pronostico = fecha_corte + 1
            fin_pronostico = inicio_pronostico + lags - 1

            if len(serie_corte) >= 10:
                try:
                    modelo_hw = ExponentialSmoothing(serie_corte, trend='add', seasonal=None).fit()
                    forecast_hw = modelo_hw.forecast(lags)
                    forecast_hw.index = range(inicio_pronostico, fin_pronostico + 1)

                    modelo_hw_13 = ExponentialSmoothing(
                        serie_corte, trend='add', seasonal='add', seasonal_periods=13
                    ).fit()
                    forecast_hw_13 = modelo_hw_13.forecast(lags)
                    forecast_hw_13.index = range(inicio_pronostico, fin_pronostico + 1)
                except:
                    forecast_hw = pd.Series([np.nan] * lags, index=range(inicio_pronostico, fin_pronostico + 1))
                    forecast_hw_13 = pd.Series([np.nan] * lags, index=range(inicio_pronostico, fin_pronostico + 1))
            else:
                forecast_hw = pd.Series([np.nan] * lags, index=range(inicio_pronostico, fin_pronostico + 1))
                forecast_hw_13 = pd.Series([np.nan] * lags, index=range(inicio_pronostico, fin_pronostico + 1))

            serie_corte.index = indice_real

            pm_3 = serie_corte.rolling(3).mean().iloc[-1] if len(serie_corte) >= 3 else np.nan
            pm_6 = serie_corte.rolling(6).mean().iloc[-1] if len(serie_corte) >= 6 else np.nan
            pm_12 = serie_corte.rolling(12).mean().iloc[-1] if len(serie_corte) >= 12 else np.nan

            pm_3_series = pd.Series([pm_3] * lags, index=range(inicio_pronostico, fin_pronostico + 1))
            pm_6_series = pd.Series([pm_6] * lags, index=range(inicio_pronostico, fin_pronostico + 1))
            pm_12_series = pd.Series([pm_12] * lags, index=range(inicio_pronostico, fin_pronostico + 1))

            demanda_real = serie.loc[inicio_pronostico:fin_pronostico]

            df_comb = pd.DataFrame({
                'real': demanda_real,
                'hw': forecast_hw,
                'hw_13': forecast_hw_13,
                'pm_3': pm_3_series,
                'pm_6': pm_6_series,
                'pm_12': pm_12_series,
            })

            if j < len(rango_turnos) - 1:
                df_comb = df_comb.dropna(subset=['real'])
                resultados_hw.append(df_comb[['real', 'hw']])
                resultados_hw_13.append(df_comb[['real', 'hw_13']])
                resultados_pm_3.append(df_comb[['real', 'pm_3']])
                resultados_pm_6.append(df_comb[['real', 'pm_6']])
                resultados_pm_12.append(df_comb[['real', 'pm_12']])
            else:
                pronostico_final_hw = df_comb[['real', 'hw']]
                pronostico_final_hw_13 = df_comb[['real', 'hw_13']]
                pronostico_final_pm_3 = df_comb[['real', 'pm_3']]
                pronostico_final_pm_6 = df_comb[['real', 'pm_6']]
                pronostico_final_pm_12 = df_comb[['real', 'pm_12']]

        modelos = {
            'hw': (resultados_hw, pronostico_final_hw),
            'hw_13': (resultados_hw_13, pronostico_final_hw_13),
            'pm_3': (resultados_pm_3, pronostico_final_pm_3),
            'pm_6': (resultados_pm_6, pronostico_final_pm_6),
            'pm_12': (resultados_pm_12, pronostico_final_pm_12),
        }

        metricas_modelos = {}
        for nombre_modelo, (resultados, _) in modelos.items():
            if resultados:
                df_resultado = pd.concat(resultados)
                df_resultado["error"] = df_resultado["real"] - df_resultado[nombre_modelo]
                df_resultado["error_abs"] = df_resultado["error"].abs()
                suma_real = df_resultado["real"].sum()
                mae_porc = df_resultado["error_abs"].sum() / suma_real
                sesgo_porc = df_resultado["error"].sum() / suma_real
                score_porc = mae_porc + abs(sesgo_porc)
                rmse = np.sqrt((df_resultado["error"] ** 2).mean())
            else:
                mae_porc = np.nan
                sesgo_porc = np.nan
                score_porc = np.inf
                rmse = np.nan

            metricas_modelos[nombre_modelo] = {
                "mae_porc": mae_porc,
                "sesgo_porc": sesgo_porc,
                "score_porc": round(score_porc, 3),
                "rmse": rmse
            }

        df_metricas = pd.DataFrame(metricas_modelos).T.sort_values("score_porc")
        mejor_modelo = df_metricas.index[0]
        pronostico_final = modelos[mejor_modelo][1]

        resultados_por_serie[clave] = {
            "mejor_modelo": mejor_modelo,
            "metricas": df_metricas,
            "pronostico_final": pronostico_final
        }

    return resultados_por_serie

# %% [markdown]
# ## Funciones de Ayuda para la generacion de reportes

# %% [markdown]
# ### Reporte Producto Terminado

# %%
def generar_resumen_pt(resultados_pt):
    """
    Genera un DataFrame resumen con los mejores modelos y pronósticos finales   
    """

    resumen_filas = []

    for (regional, producto), datos in resultados_pt.items():
        mejor_modelo = datos['mejor_modelo']
        metricas = datos['metricas']
        pronostico_final = datos['pronostico_final']

        rmse_val = metricas.loc[mejor_modelo, 'rmse']
        score_val = metricas.loc[mejor_modelo, 'score_porc']
        pronostico = pronostico_final[mejor_modelo]

        fila = {
            'REGIONAL': regional,
            'PRODUCTO': producto,
            'MODELO': mejor_modelo.upper(),  
            'SCORE_PORC': f"{round(score_val * 100, 1)}%",                     
            'RMSE': round(rmse_val, 1),
        }

        for turno, valor in pronostico.items():
            fila[turno] = round(valor, 0) if pd.notna(valor) else np.nan

        resumen_filas.append(fila)

    df_resumen = pd.DataFrame(resumen_filas)

    cols_fijas = ['REGIONAL', 'PRODUCTO','MODELO', 'SCORE_PORC', 'RMSE']
    cols_turnos = sorted([col for col in df_resumen.columns if isinstance(col, (int, str)) and col not in cols_fijas])
    df_resumen = df_resumen[cols_fijas + cols_turnos]

    return df_resumen

# %% [markdown]
# ### Reporte Materia Prima

# %%
def generar_resumen_mp(resultados_mp):
    """
    Genera un DataFrame resumen con los mejores modelos y pronósticos finales para materias primas
    """

    resumen_filas = []

    for producto, datos in resultados_mp.items():
        mejor_modelo = datos['mejor_modelo']
        metricas = datos['metricas']
        pronostico_final = datos['pronostico_final']

        rmse_val = metricas.loc[mejor_modelo, 'rmse']
        score_val = metricas.loc[mejor_modelo, 'score_porc']
        pronostico = pronostico_final[mejor_modelo]

        fila = {
            'PRODUCTO': producto,
            'MODELO': mejor_modelo.upper(),
            'SCORE_PORC': f"{round(score_val * 100, 1)}%",
            'RMSE': round(rmse_val, 1),
        }

        for turno, valor in pronostico.items():
            fila[turno] = round(valor, 0) if pd.notna(valor) else np.nan

        resumen_filas.append(fila)

    df_resumen = pd.DataFrame(resumen_filas)

    cols_fijas = ['PRODUCTO', 'MODELO', 'SCORE_PORC', 'RMSE']
    cols_turnos = sorted([col for col in df_resumen.columns if col not in cols_fijas])
    df_resumen = df_resumen[cols_fijas + cols_turnos]

    return df_resumen

# %% [markdown]
# # Script de Ejecución Parte 1 - Producto Terminado

# %% [markdown]
# # Define la carpeta donde están los archivos
# ruta_demandas = 'dataset/'
# df_agregado = cargar_demandas(ruta_demandas)
# 
# # Define los productos a considerar
# productos = ['MOTO', 'CUATRIMOTO', 'TRACTOR']
# 
# # Preprocesar los datos parte 1
# df_final = preprocesar_datos_parte_1(df_agregado, productos)
# 
# # Preprocesar los datos parte 2
# df = preprocesar_datos_parte_2(df_final)
# 
# # Mostrar el DataFrame final
# df
# 
# # Definir cololres para los productos terminados
# colores_pt = {
#     'MOTO': 'salmon',
#     'CUATRIMOTO': 'navy',
#     'TRACTOR': 'darkcyan'
# }
# # Graficar la demanda de producto terminado
# #graficar_demanda_pt(df, colores_pt)
# 
# # Crear diccionario con series de tiempo por producto y regional
# series_dict_pt = crear_dicc_pt(df)
# 
# # Realizar pronósticos para las series de tiempo de producto terminado
# resultados_pt = crear_pronosticos_generico(series_dict_pt, periodos_atras=48, lags=6)
# 
# # Graficar los pronósticos de producto terminado
# graficar_pronosticos_pt(df, resultados_pt, colores_pt)
# 
# # Generar resumen de los resultados de pronósticos de producto terminado
# df_resumen = generar_resumen_pt(resultados_pt)
# display(df_resumen)

# %% [markdown]
# # Script de Ejecución Parte 2 - Materia Prima

# %% [markdown]
# # Cargar todas las hojas del archivo Excel
# ruta_data_maestra = r'dataset\INFO_MAESTRA_BOM_TIEMPOS.xlsx'
# 
# # Cargar los DataFrames de la data maestra
# df_bom_mp, df_m_d_o, df_transporte, df_almacenamiento = cargar_data_maestra(ruta_data_maestra)
# 
# # Preprocesar los datos de materia prima
# df_bom_vertical = preprocesar_datos_mp(df_bom_mp)
# 
# # Explosionar los datos de materia prima
# df_consumo = explosionar_mp(df, df_bom_vertical)
# 
# # Crear un diccionario con series de tiempo de materia prima
# series_dict_mp = crear_dicc_mp(df_consumo)
# 
# # Generar pronósticos para las series de tiempo de materia prima
# resultados_mp = crear_pronosticos_generico(series_dict_mp, periodos_atras=48, lags=12)
# 
# # Generar colores para las materias primas
# colores_mp = generar_colores_mp(df_bom_vertical['MATERIA_PRIMA'].unique())
# 
# # Graficar los pronósticos de materia prima
# graficar_pronosticos_mp(df_consumo, resultados_mp, colores_mp, lags=12)
# 
# # Generar resumen de los resultados de pronósticos de materia prima
# generar_resumen_mp(resultados_mp)
# 
# 

# %% [markdown]
# # Front End Streamlit

# %%

st.set_page_config(page_title="App de Pronósticos Mototrak", layout="wide")

st.title("App de Pronósticos para Producto Terminado y Materia Prima")
#pestaña_pt, pestaña_mp = st.tabs(["Pronósticos PT", "Pronósticos MP"])
seccion = st.sidebar.radio("Selecciona sección", ["Pronósticos PT", "Pronósticos MP"])
# ----------------------------
# PESTAÑA PRODUCTO TERMINADO
# ----------------------------
if seccion == "Pronósticos PT":
    st.subheader("Cargar archivos de demanda por región")
    archivo_norte = st.file_uploader("Archivo demanda NORTE", type=["csv"])
    archivo_centro = st.file_uploader("Archivo demanda CENTRO", type=["csv"])
    archivo_sur = st.file_uploader("Archivo demanda SUR", type=["csv"])

    periodos_atras_pt = st.number_input("Periodos hacia atrás para backtesting (PT)", min_value=1, max_value=60, value=12)
    lags_pt = st.number_input("Cantidad de periodos a pronosticar (lags PT)", min_value=1, max_value=24, value=6)

    if archivo_norte and archivo_centro and archivo_sur:
        productos = ['MOTO', 'CUATRIMOTO', 'TRACTOR']
        df_agregado = cargar_demandas_por_region(archivo_norte, archivo_centro, archivo_sur)
        df_final = preprocesar_datos_parte_1(df_agregado, productos)
        df = preprocesar_datos_parte_2(df_final)
        st.session_state["df"] = df

        colores_pt = {'MOTO': 'salmon', 'CUATRIMOTO': 'navy', 'TRACTOR': 'darkcyan'}
        series_dict_pt = crear_dicc_pt(df)

        if st.button("Generar pronóstico de PT"):
          
            resultados_pt = crear_pronosticos_generico(series_dict_pt, periodos_atras_pt, lags_pt)
            df_resumen_pt = generar_resumen_pt(resultados_pt)

            st.session_state['resultados_pt'] = resultados_pt
            st.session_state['df_resumen_pt'] = df_resumen_pt

            fig = graficar_pronosticos_pt(df, resultados_pt, colores_pt)
            st.session_state['fig_pt'] = fig

    # Mostrar resultados si ya existen
    if 'df_resumen_pt' in st.session_state:
        st.subheader("Resumen del pronóstico PT")
        st.dataframe(st.session_state['df_resumen_pt'], use_container_width=True)

        # Reconstruir gráfica si no está en session_state
        if 'fig_pt' not in st.session_state:
            st.session_state['fig_pt'] = graficar_pronosticos_pt(
                st.session_state['df'],
                st.session_state['resultados_pt'],
                {'MOTO': 'salmon', 'CUATRIMOTO': 'navy', 'TRACTOR': 'darkcyan'}
            )

        st.plotly_chart(st.session_state['fig_pt'], use_container_width=True)

        buffer_pt = io.BytesIO()
        st.session_state['df_resumen_pt'].to_excel(buffer_pt, index=False)
        st.download_button(
            "📥 Descargar resumen PT en Excel",
            data=buffer_pt.getvalue(),
            file_name="resumen_pt.xlsx"
        )

# ----------------------------
# PESTAÑA MATERIA PRIMA
# ----------------------------
elif seccion == "Pronósticos MP":
    st.subheader("Cargar archivo maestro de datos")
    archivo_maestro = st.file_uploader("Archivo Excel (Info Maestra)", type=["xlsx"])

    if archivo_maestro:
        df_bom_mp, df_m_d_o, df_transporte, df_almacenamiento = cargar_data_maestra(archivo_maestro)
        df_bom_vertical = preprocesar_datos_mp(df_bom_mp)
        st.session_state["df_bom_vertical"] = df_bom_vertical  # 💾 Guardar en session_state

        if st.button("Ejecutar explosión de materiales"):
            try:
                if "df" not in st.session_state:
                    st.warning("Primero debes generar el pronóstico de Producto Terminado.")
                    st.stop()

                df = st.session_state["df"]
                df_consumo = explosionar_mp(df, df_bom_vertical)
                st.session_state["df_consumo"] = df_consumo  # 💾 Guardar en session_state
                st.success("Explosión realizada con éxito")
            except Exception as e:
                st.error(f"Error durante la explosión de materiales: {e}")

        # Parámetros visibles siempre que haya datos disponibles
        if "df_consumo" in st.session_state and "df_bom_vertical" in st.session_state:
            periodos_atras_mp = st.number_input("Periodos hacia atrás para backtesting (MP)", min_value=1, max_value=60, value=12)
            lags_mp = st.number_input("Cantidad de periodos a pronosticar (lags MP)", min_value=1, max_value=24, value=6)

            if st.button("Generar pronóstico de MP"):
                try:
                    df_consumo = st.session_state["df_consumo"]
                    df_bom_vertical = st.session_state["df_bom_vertical"]

                    series_dict_mp = crear_dicc_mp(df_consumo)                  
                    resultados_mp = crear_pronosticos_generico(series_dict_mp, periodos_atras_mp, lags_mp)
                    df_resumen_mp = generar_resumen_mp(resultados_mp)

                    st.session_state['resultados_mp'] = resultados_mp
                    st.session_state['df_resumen_mp'] = df_resumen_mp

                    colores_mp = generar_colores_mp(df_bom_vertical['MATERIA_PRIMA'].unique())
                    st.session_state['colores_mp'] = colores_mp

                    fig = graficar_pronosticos_mp(df_consumo, resultados_mp, colores_mp, lags=lags_mp)
                    st.session_state['fig_mp'] = fig
                    #st.dataframe(df_resumen_mp, use_container_width=True)



                except Exception as e:
                    st.error(f"Error durante el pronóstico: {e}")

    # Mostrar resultados si ya existen
    if 'df_resumen_mp' in st.session_state:
        st.subheader("Resumen del pronóstico MP")
        st.dataframe(st.session_state['df_resumen_mp'], use_container_width=True)

        # Reconstruir gráfica si no está en session_state
        if 'fig_mp' not in st.session_state:
            st.session_state['fig_mp'] = graficar_pronosticos_mp(
                st.session_state['df'],
                st.session_state['resultados_mp'],
                st.session_state['colores_mp']
            )

        st.plotly_chart(st.session_state['fig_mp'], use_container_width=True)

        buffer_mp = io.BytesIO()
        st.session_state['df_resumen_mp'].to_excel(buffer_mp, index=False)
        st.download_button(
            "📥 Descargar resumen MP en Excel",
            data=buffer_mp.getvalue(),
            file_name="resumen_mp.xlsx"
        )

