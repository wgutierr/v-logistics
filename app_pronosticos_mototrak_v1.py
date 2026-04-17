# %%
# Funciones basicas
import pandas as pd
import numpy as np



# Funciones de graficaciÃ³n
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Funciones de pronÃ³stico y estadisticas
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
            regional, anio = filename.split('_')
            anio = anio.replace('.csv', '')

            

            # Carga el archivo y aÃ±ade las columnas "Producto" y "Regional"
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
    Carga y concatena los archivos de demanda por regiÃ³n (NORTE, CENTRO, SUR) 
    desde archivos subidos vÃ­a Streamlit.
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
            st.write(f"Ultimo turno cargado para {region}: {df['Turn'].max()}")

            dataframes.append(df)



    df_agregado = pd.concat(dataframes, ignore_index=True)



    return df_agregado





def _normalizar_nombre_regional(nombre):
    """
    Normaliza etiquetas de regional a NORTE, CENTRO o SUR.
    """
    if pd.isna(nombre):
        return None



    texto = str(nombre).strip().upper()
    if "NORTE" in texto:
        return "NORTE"
    if "CENTRO" in texto:
        return "CENTRO"
    if "SUR" in texto:
        return "SUR"
    return None





def cargar_demandas_desde_excel_demanda(archivo_excel, hoja='Demand'):
    """
    Carga la demanda de PT desde un Ãºnico Excel (hoja 'Demand').



    Soporta dos estructuras:
    1) Formato tabular con columnas REGIONAL, Turn, MOTO, CUATRIMOTO, TRACTOR.
    2) Formato en bloques por regional (Sur/Centro/Norte) con dos filas de encabezado.
    """



    # Intento 1: formato tabular directo.
    df_tabular = pd.read_excel(archivo_excel, sheet_name=hoja)
    columnas_tabular = {str(c).strip().upper(): c for c in df_tabular.columns}
    if 'REGIONAL' in columnas_tabular and 'TURN' in columnas_tabular:
        df_tabular = df_tabular.rename(columns={v: k for k, v in columnas_tabular.items()})
        df_tabular['REGIONAL'] = df_tabular['REGIONAL'].astype(str).str.upper().str.strip()
        st.write(

            "Ultimo turno cargado por regional:",
            df_tabular.groupby('REGIONAL')['TURN'].max().to_dict()
        )
        # Mantener nombre de columna esperado en el resto del pipeline.
        return df_tabular.rename(columns={'TURN': 'Turn'})



    # Intento 2: formato por bloques regionales.
    archivo_excel.seek(0)
    df_raw = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)



    if df_raw.shape[0] < 3:
        raise ValueError("La hoja 'Demand' no tiene el formato esperado.")



    fila_regionales = df_raw.iloc[0]
    fila_columnas = df_raw.iloc[1]



    turn_cols = [
        i for i, val in enumerate(fila_columnas)
        if str(val).strip().upper() == 'TURN'
    ]



    if not turn_cols:
        raise ValueError("No se encontraron columnas 'Turn' en la hoja 'Demand'.")



    dataframes = []
    n_cols = df_raw.shape[1]



    for idx, col_inicio in enumerate(turn_cols):
        col_fin = turn_cols[idx + 1] if idx + 1 < len(turn_cols) else n_cols



        regional = _normalizar_nombre_regional(fila_regionales.iloc[col_inicio])
        if regional is None:
            continue



        encabezados = [
            str(c).strip().upper()
            for c in fila_columnas.iloc[col_inicio:col_fin].tolist()
        ]
        bloque = df_raw.iloc[2:, col_inicio:col_fin].copy()
        bloque.columns = encabezados



        if 'TURN' not in bloque.columns:
            continue



        bloque = bloque.dropna(subset=['TURN'])
        bloque['TURN'] = pd.to_numeric(bloque['TURN'], errors='coerce')
        bloque = bloque.dropna(subset=['TURN'])
        bloque['TURN'] = bloque['TURN'].astype(int)



        for col in ['MOTO', 'CUATRIMOTO', 'TRACTOR']:
            if col in bloque.columns:
                bloque[col] = pd.to_numeric(bloque[col], errors='coerce')



        bloque['REGIONAL'] = regional
        dataframes.append(bloque)
        st.write(f"Ultimo turno cargado para {regional}: {bloque['TURN'].max()}")




    if not dataframes:
        raise ValueError("No se pudo extraer demanda por regional desde la hoja 'Demand'.")



    df_agregado = pd.concat(dataframes, ignore_index=True)
    return df_agregado.rename(columns={'TURN': 'Turn'})





def cargar_bom_desde_excel_gameplay(archivo_excel, hoja='Products & Materials'):
    """
    Extrae la matriz BOM (PRODUCTO x MOTO/CUATRIMOTO/TRACTOR) desde
    la hoja 'Products & Materials' del archivo gameplay-export.
    """
    archivo_excel.seek(0)
    raw = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)



    # Buscar la fila de encabezado de la relaciÃ³n Product - Raw Material.
    header_idx = None
    for i in range(len(raw)):
        val = str(raw.iloc[i, 0]).strip().upper() if pd.notna(raw.iloc[i, 0]) else ""
        if "PRODUCT \\ RAW MATERIAL" in val:
            header_idx = i
            break



    if header_idx is None:
        raise ValueError("No se encontró la tabla 'Product - Raw Material Relationship'.")



    header = raw.iloc[header_idx]
    col_indices = [j for j in range(1, len(header)) if pd.notna(header.iloc[j])]
    if not col_indices:
        raise ValueError("No se encontraron materias primas en 'Products & Materials'.")



    # Filas esperadas inmediatamente despuÃ©s del encabezado.
    row_moto = raw.iloc[header_idx + 1]
    row_cuatri = raw.iloc[header_idx + 2]
    row_tractor = raw.iloc[header_idx + 3]



    nombre_moto = str(row_moto.iloc[0]).strip().upper() if pd.notna(row_moto.iloc[0]) else ""
    nombre_cuatri = str(row_cuatri.iloc[0]).strip().upper() if pd.notna(row_cuatri.iloc[0]) else ""
    nombre_tractor = str(row_tractor.iloc[0]).strip().upper() if pd.notna(row_tractor.iloc[0]) else ""
    if not ("MOTO" in nombre_moto and "CUATRIMOTO" in nombre_cuatri and "TRACTOR" in nombre_tractor):
        raise ValueError("Las filas de MOTO/CUATRIMOTO/TRACTOR no tienen el formato esperado.")



    filas = []
    for j in col_indices:
        filas.append({
            'PRODUCTO': str(header.iloc[j]).strip(),
            'MOTO': 0 if pd.isna(row_moto.iloc[j]) else row_moto.iloc[j],
            'CUATRIMOTO': 0 if pd.isna(row_cuatri.iloc[j]) else row_cuatri.iloc[j],
            'TRACTOR': 0 if pd.isna(row_tractor.iloc[j]) else row_tractor.iloc[j],
        })



    df_bom_mp = pd.DataFrame(filas)
    for col in ['MOTO', 'CUATRIMOTO', 'TRACTOR']:
        df_bom_mp[col] = pd.to_numeric(df_bom_mp[col], errors='coerce').fillna(0)



    return df_bom_mp





# %% [markdown]
# ## Carga de datos maestros



# %%
def cargar_data_maestra(ruta_data_maestra):



    # Carga todas las hojas como diccionario
    hojas = pd.read_excel(ruta_data_maestra, sheet_name=None)  



    # Crear un DataFrame por cada hoja con nombre df_{nombre_hoja}
    for nombre_hoja, df in hojas.items():
        # Limpiar y estandarizar el nombre de la hoja
        nombre_limpio = re.sub(r'\W+', '_', nombre_hoja.lower())  # MinÃºsculas y reemplazo de no alfanumÃ©ricos por "_"
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
    Devuelve el DataFrame final listo para anÃ¡lisis o modelado.
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
        value_vars=['MOTO', 'CUATRIMOTO', 'TRACTOR'],  # Columnas que se convertirÃ¡n en filas
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
# ### Materia Prima - ExplosiÃ³n de Materiales



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
# # Funciones de Ayuda para GrÃ¡ficas de Demanda



# %% [markdown]
# ## FunciÃ³n para graficar la demanda del producto terminado



# %%
def graficar_demanda_pt(df, colores_pt):



    """
    Crea un gráfico de lÃ­neas para la demanda de productos terminados
    por regionales y productos, utilizando Plotly.
    """



    # Listas de regionales y productos Ãºnicos
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
        showlegend_flag = False  # Solo en el primer grÃ¡fico



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



    # Mostrar grÃ¡fico
    fig.show()



# %% [markdown]
# ## FunciÃ³n para graficar los pronÃ³sticos de productos terminados



# %%
def graficar_pronosticos_pt(df, resultados_pt, colores_pt):



    # Listas de regionales y productos Ãºnicos
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



    # Agregar trazos de demanda real y pronÃ³stico
    for region in regionales:
        row, col = subplot_pos[region]
        df_region = df[df['REGIONAL'] == region]



        for producto in productos:
            # 1. Demanda real
            df_sub = df_region[df_region['PRODUCTO'] == producto]
            clave = (region, producto)



            if not df_sub.empty:
                # Si hay pronÃ³stico, ajustar la longitud del histÃ³rico
                if clave in resultados_pt:
                    pronostico_final = resultados_pt[clave]["pronostico_final"]
                    mejor_modelo = resultados_pt[clave]["mejor_modelo"]



                    lags = len(pronostico_final)  # Cantidad de pasos de pronÃ³stico
                    df_sub = df_sub.tail(52 + lags)  # Cortar a los Ãºltimos 52 + lags



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
                    # Si no hay pronÃ³stico, igual limitar a Ãºltimos 52 datos
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
# ## FunciÃ³n para graficar los pronosticos de materia prima



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
    Grafica series de consumo real y pronÃ³stico para materias primas.



    Parámetros:
    - df: DataFrame con columnas ['Turn', 'MATERIA_PRIMA', 'DEMANDA_MATERIA_PRIMA']
    - resultados_por_serie: dict con claves = materia prima y valores con 'pronostico_final' y 'mejor_modelo'
    - colores_mp: dict {materia_prima: color}
    - lags: nÃºmero de pasos de pronÃ³stico
    """



    elementos = sorted(df['MATERIA_PRIMA'].unique())
    n = len(elementos)
    cols = 3
    rows = math.ceil(n / cols)



    # Dividir nombres largos con salto de lÃ­nea si exceden cierto nÃºmero de caracteres
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





def construir_pronostico_pt_planta(resultados_pt):
    """
    Construye un DataFrame largo con pronóstico de unidades por turno y producto
    para planta (prioriza REGIONAL='MOTOTRAK').
    """
    productos = ['MOTO', 'CUATRIMOTO', 'TRACTOR']
    filas = []



    for producto in productos:
        clave_mototrak = ('MOTOTRAK', producto)
        if clave_mototrak in resultados_pt:
            datos = resultados_pt[clave_mototrak]
            mejor_modelo = datos['mejor_modelo']
            pron = datos['pronostico_final'][mejor_modelo]
            for turno, valor in pron.items():
                if pd.notna(valor):
                    filas.append({
                        'Turn': int(turno),
                        'PRODUCTO': producto,
                        'UNIDADES_PRONOSTICO': float(valor)
                    })
        else:
            # Respaldo si no existiera MOTOTRAK: suma NORTE + CENTRO + SUR.
            series_sum = []
            for regional in ['NORTE', 'CENTRO', 'SUR']:
                clave = (regional, producto)
                if clave in resultados_pt:
                    datos = resultados_pt[clave]
                    mejor_modelo = datos['mejor_modelo']
                    s = datos['pronostico_final'][mejor_modelo]
                    series_sum.append(s)
            if series_sum:
                pron = pd.concat(series_sum, axis=1).sum(axis=1, min_count=1)
                for turno, valor in pron.items():
                    if pd.notna(valor):
                        filas.append({
                            'Turn': int(turno),
                            'PRODUCTO': producto,
                            'UNIDADES_PRONOSTICO': float(valor)
                        })



    return pd.DataFrame(filas)





def calcular_carga_procesos_planta(df_pron_pt, df_tiempos_prod, df_capacidad_proc):
    """
    Convierte pronóstico de PT a minutos requeridos por proceso y turno.
    """
    if df_pron_pt.empty:
        return pd.DataFrame(columns=['Turn', 'PROCESO', 'MIN_REQUERIDOS', 'CAPACIDAD_MIN'])



    base = df_pron_pt.merge(df_tiempos_prod, on='PRODUCTO', how='left')
    base['MIN_REQUERIDOS'] = base['UNIDADES_PRONOSTICO'] * base['MIN_POR_UNIDAD']



    carga = (
        base.groupby(['Turn', 'PROCESO'], as_index=False)['MIN_REQUERIDOS']
        .sum()
        .merge(df_capacidad_proc, on='PROCESO', how='left')
    )
    return carga





def graficar_capacidad_procesos(df_carga):
    """
    Grafica carga (min requeridos) vs capacidad (línea horizontal por proceso).
    """
    procesos = sorted(df_carga['PROCESO'].dropna().unique().tolist())
    if not procesos:
        return go.Figure()



    cols = 2
    rows = math.ceil(len(procesos) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=procesos)



    for i, proc in enumerate(procesos):
        row = (i // cols) + 1
        col = (i % cols) + 1
        sub = df_carga[df_carga['PROCESO'] == proc].sort_values('Turn')
        cap = sub['CAPACIDAD_MIN'].dropna().iloc[0] if not sub['CAPACIDAD_MIN'].dropna().empty else 0



        fig.add_trace(
            go.Scatter(
                x=sub['Turn'],
                y=sub['MIN_REQUERIDOS'],
                mode='lines+markers',
                name=f'{proc} - requeridos',
                line=dict(color='royalblue')
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=sub['Turn'],
                y=[cap] * len(sub),
                mode='lines',
                name=f'{proc} - capacidad',
                line=dict(color='firebrick', dash='dash')
            ),
            row=row, col=col
        )



    fig.update_layout(
        height=350 * rows,
        width=1200,
        title_text='Capacidad de Procesos de Planta (minutos por turno)',
        showlegend=False,
        template='ggplot2'
    )
    fig.update_xaxes(title_text='Turn')
    fig.update_yaxes(title_text='Minutos')
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
# ## CreaciÃ³n de diccionario con series de tiempo materia prima en mototrak



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
# Se hará backtesting desde n periodos hacia atras y generando mÃºltiples pronÃ³sticos hacia adelante (lags)



# %%
def crear_pronosticos_generico_legacy(series_dict, periodos_atras=48, lags=6, modelos_seleccionados=None):
    """
    Implementación original de backtesting por serie (fallback).
    """
    turnos = next(iter(series_dict.values())).index.tolist()
    rango_turnos = turnos[-(periodos_atras + 1):]
    resultados_por_serie = {}



    progreso = st.empty() if USANDO_STREAMLIT else None
    barra = st.progress(0) if USANDO_STREAMLIT else None
    total = len(series_dict)



    for i, (clave, serie) in enumerate(series_dict.items()):
        if USANDO_STREAMLIT:
            progreso.markdown(f"ð¨âð» Analizando `{clave}` (fallback legacy)...")
            barra.progress((i + 1) / total)
        else:
            print(f"Analizando {clave} (fallback legacy)")



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
                except Exception:
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





# %%
def crear_pronosticos_generico(series_dict, periodos_atras=48, lags=6, modelos_seleccionados=None):
    """
    Aplica validación cruzada temporal global (una sola corrida para todas las series)
    usando StatsForecast (Nixtla).
    Funciona tanto para productos terminados como materias primas.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            Holt,
            HoltWinters,
            WindowAverage,
            MSTL,
            SimpleExponentialSmoothingOptimized,
            AutoARIMA,
            AutoTheta,
            DynamicOptimizedTheta
        )
    except Exception as e:
        if USANDO_STREAMLIT:
            st.warning(
                f"No se pudo importar StatsForecast/Nixtla ({e}). "
                "Se usará el método legacy por serie."
            )
        return crear_pronosticos_generico_legacy(
            series_dict,
            periodos_atras=periodos_atras,
            lags=lags,
            modelos_seleccionados=modelos_seleccionados
        )



    if not series_dict:
        return {}



    progreso = st.empty() if USANDO_STREAMLIT else None
    barra = st.progress(0) if USANDO_STREAMLIT else None
    if USANDO_STREAMLIT:
        progreso.markdown("🔄Preparando series para CV global...")
        barra.progress(0.1)
    else:
        print("👨‍💻Preparando series para CV global...")



    # Empaquetar todas las series en formato largo para StatsForecast.
    frames = []
    id_to_clave = {}
    claves_ordenadas = list(series_dict.keys())
    for i, clave in enumerate(claves_ordenadas):
        serie = series_dict[clave].dropna().sort_index()
        if serie.empty:
            continue



        uid = f"s_{i}"
        id_to_clave[uid] = clave
        frame = pd.DataFrame({
            'unique_id': uid,
            'ds': pd.to_numeric(serie.index, errors='coerce'),
            'y': pd.to_numeric(serie.values, errors='coerce')
        }).dropna(subset=['ds', 'y'])
        if frame.empty:
            continue
        frame['ds'] = frame['ds'].astype(int)
        frames.append(frame)



    if not frames:
        return {}



    df_sf = pd.concat(frames, ignore_index=True).sort_values(['unique_id', 'ds'])



    modelos_default = ['hw', 'hw_13', 'wa_3', 'wa_6', 'wa_12', 'ses', 'mstl']
    modelos_ids = [m.lower() for m in (modelos_seleccionados or modelos_default)]

    builders = {
        'hw': lambda: Holt(alias='hw'),
        'hw_13': lambda: HoltWinters(season_length=13, alias='hw_13'),
        'wa_3': lambda: WindowAverage(window_size=3, alias='wa_3'),
        'wa_6': lambda: WindowAverage(window_size=6, alias='wa_6'),
        'wa_12': lambda: WindowAverage(window_size=12, alias='wa_12'),
        'ses': lambda: SimpleExponentialSmoothingOptimized(alias='ses'),
        'mstl': lambda: MSTL(season_length=13, alias='mstl'),
        # Se aproxima SARIMAX con AutoARIMA estacional.
        'autoarima': lambda: AutoARIMA(season_length=1, alias='autoarima'),
        'sarimax': lambda: AutoARIMA(season_length=13, alias='sarimax'),
        'theta': lambda: AutoTheta(season_length=13, alias='theta'),
        'doc': lambda: DynamicOptimizedTheta(season_length=13, alias='doc'),
    }

    modelos = []
    nombres_modelos = []
    for m in modelos_ids:
        if m in builders:
            modelos.append(builders[m]())
            nombres_modelos.append(m)

    if not modelos:
        raise ValueError("No hay modelos válidos seleccionados para pronóstico.")



    if USANDO_STREAMLIT:
        progreso.markdown("🔄 Ejecutando cross_validation global con Nixtla...")
        barra.progress(0.35)
    else:
        print("Ejecutando cross_validation global con Nixtla...")



    tamanos = df_sf.groupby('unique_id').size()
    min_len = int(tamanos.min())
    # Heurí­stica para evitar fallo por 'tiny datasets' en CV.
    n_windows_cv = max(1, min(int(periodos_atras), max(1, min_len - lags - 8)))



    sf = StatsForecast(models=modelos, freq=1, n_jobs=-1)



    try:
        cv_df = sf.cross_validation(
            df=df_sf,
            h=int(lags),
            n_windows=int(n_windows_cv),
            step_size=1
        )
    except Exception:
        # Fallback global sin modelos estacionales si hay series cortas.
        fallback_ids = [m for m in nombres_modelos if m in {'hw', 'wa_3', 'wa_6', 'wa_12', 'ses', 'autoarima'}]
        if not fallback_ids:
            fallback_ids = ['hw']
        modelos_fallback = [builders[m]() for m in fallback_ids]
        nombres_modelos = fallback_ids
        sf = StatsForecast(models=modelos_fallback, freq=1, n_jobs=-1)
        cv_df = sf.cross_validation(
            df=df_sf,
            h=int(lags),
            n_windows=int(n_windows_cv),
            step_size=1
        )



    if USANDO_STREAMLIT:
        progreso.markdown("📈 Generando pronóstico final global...")
        barra.progress(0.75)
    else:
        print("Generando pronóstico final global...")



    forecast_df = sf.forecast(df=df_sf, h=int(lags))
    forecast_cols = [c for c in forecast_df.columns if c not in ['unique_id', 'ds']]
    nombres_modelos = [m for m in nombres_modelos if m in forecast_cols]



    resultados_por_serie = {}



    for uid, clave in id_to_clave.items():
        serie_original = series_dict[clave].sort_index()
        cv_uid = cv_df[cv_df['unique_id'] == uid].copy()
        fc_uid = forecast_df[forecast_df['unique_id'] == uid].copy()



        metricas_modelos = {}
        for nombre_modelo in nombres_modelos:
            if nombre_modelo not in cv_uid.columns:
                mae_porc = np.nan
                sesgo_porc = np.nan
                score_porc = np.inf
                rmse = np.nan
            else:
                df_resultado = cv_uid[['y', nombre_modelo]].dropna().copy()
                if df_resultado.empty:
                    mae_porc = np.nan
                    sesgo_porc = np.nan
                    score_porc = np.inf
                    rmse = np.nan
                else:
                    df_resultado["error"] = df_resultado["y"] - df_resultado[nombre_modelo]
                    df_resultado["error_abs"] = df_resultado["error"].abs()
                    suma_real = df_resultado["y"].sum()
                    if suma_real == 0:
                        mae_porc = np.nan
                        sesgo_porc = np.nan
                        score_porc = np.inf
                    else:
                        mae_porc = df_resultado["error_abs"].sum() / suma_real
                        sesgo_porc = df_resultado["error"].sum() / suma_real
                        score_porc = mae_porc + abs(sesgo_porc)
                    rmse = np.sqrt((df_resultado["error"] ** 2).mean())



            metricas_modelos[nombre_modelo] = {
                "mae_porc": mae_porc,
                "sesgo_porc": sesgo_porc,
                "score_porc": round(score_porc, 3) if np.isfinite(score_porc) else np.inf,
                "rmse": rmse
            }



        df_metricas = pd.DataFrame(metricas_modelos).T.sort_values("score_porc")
        mejor_modelo = df_metricas.index[0]



        if not fc_uid.empty and mejor_modelo in fc_uid.columns:
            fc_uid = fc_uid.sort_values('ds').set_index('ds')
            real = serie_original.reindex(fc_uid.index)
            pronostico_final = pd.DataFrame({
                'real': real,
                mejor_modelo: fc_uid[mejor_modelo]
            })
        else:
            # Fallback defensivo para no romper flujos de reporte y grÃ¡fica.
            ultimo_turno = int(serie_original.index.max())
            idx_futuro = pd.Index(range(ultimo_turno + 1, ultimo_turno + int(lags) + 1))
            pronostico_final = pd.DataFrame({
                'real': serie_original.reindex(idx_futuro),
                mejor_modelo: np.nan
            }, index=idx_futuro)



        resultados_por_serie[clave] = {
            "mejor_modelo": mejor_modelo,
            "metricas": df_metricas,
            "pronostico_final": pronostico_final
        }



    if USANDO_STREAMLIT:
        barra.progress(1.0)

        progreso.markdown("CV global y pronostico final completados.")
    else:
        print("CV global y pronóstico final completados.")



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
# # Script de EjecuciÃ³n Parte 1 - Producto Terminado



# %% [markdown]
# # Define la carpeta donde estÃ¡n los archivos
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
# # Realizar pronÃ³sticos para las series de tiempo de producto terminado
# resultados_pt = crear_pronosticos_generico(series_dict_pt, periodos_atras=48, lags=6)
# 
# # Graficar los pronÃ³sticos de producto terminado
# graficar_pronosticos_pt(df, resultados_pt, colores_pt)
# 
# # Generar resumen de los resultados de pronÃ³sticos de producto terminado
# df_resumen = generar_resumen_pt(resultados_pt)
# display(df_resumen)



# %% [markdown]
# # Script de EjecuciÃ³n Parte 2 - Materia Prima



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
# # Generar pronÃ³sticos para las series de tiempo de materia prima
# resultados_mp = crear_pronosticos_generico(series_dict_mp, periodos_atras=48, lags=12)
# 
# # Generar colores para las materias primas
# colores_mp = generar_colores_mp(df_bom_vertical['MATERIA_PRIMA'].unique())
# 
# # Graficar los pronÃ³sticos de materia prima
# graficar_pronosticos_mp(df_consumo, resultados_mp, colores_mp, lags=12)
# 
# # Generar resumen de los resultados de pronÃ³sticos de materia prima
# generar_resumen_mp(resultados_mp)
# 
# 



# %% [markdown]
# # Front End Streamlit



# %%



st.set_page_config(page_title="App de Pronósticos Mototrak", layout="wide")



st.title("App de Pronósticos para Producto Terminado y Materia Prima")
#pestaña_pt, pestaña_mp = st.tabs(["Pronósticos PT", "Pronósticos MP"])
def normalizar_nombre_proceso(nombre):
    """
    Lleva variaciones de nombres de proceso a una etiqueta canónica común
    para gameplay-export y MotoTrack.
    """
    texto = "" if pd.isna(nombre) else str(nombre).strip()
    if not texto:
        return texto

    canon = texto.upper().translate(str.maketrans("ÁÉÍÓÚÜ", "AEIOUU"))
    canon = re.sub(r"\s+", " ", canon)

    equivalencias = {
        "ARMAR BASE": "Armar Base",
        "ARMAR CHASIS": "Armar Chasis",
        "RUEDA - ORUGA": "Rueda - Oruga",
        "RUEDAS - ORUGA": "Rueda - Oruga",
        "ASIENTO - VOLANTE": "Asiento - Volante",
    }
    return equivalencias.get(canon, texto)


def cargar_produccion_desde_excel_gameplay(archivo_excel, hoja='Production'):
    """
    Extrae tiempos de proceso (min/unidad) y capacidades por proceso desde
    la hoja 'Production' del archivo gameplay-export.

    Esta versiÃ³n tolera el formato compacto actual de MotoTrak y no falla
    si la fila de disponibilidad no viene o cambia de nombre.
    """
    archivo_excel.seek(0)
    raw = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)

    header_idx = None
    for i in range(len(raw)):
        fila_texto = " | ".join(
            [str(v).strip().upper() for v in raw.iloc[i].tolist() if pd.notna(v)]
        )
        if "ARMAR BASE" in fila_texto:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No se encontrÃ³ el encabezado de procesos en 'Production'.")

    header = raw.iloc[header_idx]
    procesos = []
    col_procesos = []
    for j in range(1, len(header)):
        if pd.notna(header.iloc[j]) and str(header.iloc[j]).strip():
            procesos.append(normalizar_nombre_proceso(header.iloc[j]))
            col_procesos.append(j)
    if not procesos:
        raise ValueError("No se encontraron procesos en 'Production'.")

    productos_objetivo = {"MOTO", "CUATRIMOTO", "TRACTOR"}
    filas_producto = []
    for i in range(header_idx + 1, len(raw)):
        nombre = str(raw.iloc[i, 0]).strip().upper() if pd.notna(raw.iloc[i, 0]) else ""
        if nombre in productos_objetivo:
            filas_producto.append(i)
    if len(filas_producto) < 3:
        raise ValueError("No se encontraron las filas de MOTO/CUATRIMOTO/TRACTOR en 'Production'.")

    etiquetas_capacidad = (
        "DISPONIBILIDAD", "AVAILABILITY", "CAPACIDAD", "CAPACITY",
        "MIN DISPONIBLES", "MIN AVAILABLE", "TIEMPO DISPONIBLE"
    )
    disp_idx = None
    for i in range(header_idx + 1, len(raw)):
        fila_texto = " | ".join(
            [str(v).strip().upper() for v in raw.iloc[i].tolist() if pd.notna(v)]
        )
        if any(etiqueta in fila_texto for etiqueta in etiquetas_capacidad):
            disp_idx = i
            break

    registros_tiempo = []
    for i in filas_producto:
        producto = str(raw.iloc[i, 0]).strip().upper()
        for proc, j in zip(procesos, col_procesos):
            valor = pd.to_numeric(raw.iloc[i, j], errors='coerce')
            registros_tiempo.append({
                "PRODUCTO": producto,
                "PROCESO": proc,
                "MIN_POR_UNIDAD": 0 if pd.isna(valor) else float(valor)
            })
    df_tiempos = pd.DataFrame(registros_tiempo)

    registros_cap = []
    for proc, j in zip(procesos, col_procesos):
        valor = np.nan
        if disp_idx is not None:
            valor = pd.to_numeric(raw.iloc[disp_idx, j], errors='coerce')
        registros_cap.append({
            "PROCESO": proc,
            "CAPACIDAD_MIN": np.nan if pd.isna(valor) else float(valor)
        })
    df_capacidad = pd.DataFrame(registros_cap)

    return df_tiempos, df_capacidad


seccion = st.sidebar.radio("Selecciona sección", ["Pronósticos PT", "Pronósticos MP", "Capacidad Procesos Planta"])
modelos_disponibles = ['hw', 'hw_13', 'wa_3', 'wa_6', 'wa_12', 'ses', 'mstl', 'autoarima', 'sarimax', 'theta', 'doc']
modelos_default = ['hw', 'hw_13', 'wa_3', 'wa_6', 'wa_12', 'ses', 'mstl']
modelos_seleccionados = st.sidebar.multiselect(
    "Modelos de pronóstico",
    options=modelos_disponibles,
    default=modelos_default
)
if not modelos_seleccionados:
    st.sidebar.warning("Selecciona al menos un modelo. Se usará la selección por defecto.")
    modelos_seleccionados = modelos_default
# ----------------------------
# PESTAÑA PRODUCTO TERMINADO
# ----------------------------
if seccion == "Pronósticos PT":
    st.subheader("Cargar archivo de demanda consolidado")
    archivo_demanda = st.file_uploader(
        "Archivo Excel de demanda (hoja 'Demand')",
        type=["xlsx"]
    )



    periodos_atras_pt = st.number_input("Periodos hacia atrás para backtesting (PT)", min_value=1, max_value=60, value=12)
    lags_pt = st.number_input("Cantidad de periodos a pronosticar (lags PT)", min_value=1, max_value=24, value=6)



    if archivo_demanda:
        productos = ['MOTO', 'CUATRIMOTO', 'TRACTOR']
        df_agregado = cargar_demandas_desde_excel_demanda(archivo_demanda, hoja='Demand')
        df_final = preprocesar_datos_parte_1(df_agregado, productos)
        df = preprocesar_datos_parte_2(df_final)
        st.session_state["df"] = df
        try:
            df_bom_mp_game = cargar_bom_desde_excel_gameplay(archivo_demanda, hoja='Products & Materials')
            df_bom_vertical = preprocesar_datos_mp(df_bom_mp_game)
            st.session_state["df_bom_vertical"] = df_bom_vertical
        except Exception as e:
            st.warning(f"No se pudo cargar BOM desde 'Products & Materials': {e}")
        try:
            df_tiempos_prod, df_capacidad_proc = cargar_produccion_desde_excel_gameplay(
                archivo_demanda, hoja='Production'
            )
            st.session_state["df_tiempos_prod"] = df_tiempos_prod
            st.session_state["df_capacidad_proc"] = df_capacidad_proc
        except Exception as e:
            st.warning(f"No se pudo cargar información de 'Production': {e}")



        colores_pt = {'MOTO': 'salmon', 'CUATRIMOTO': 'navy', 'TRACTOR': 'darkcyan'}
        series_dict_pt = crear_dicc_pt(df)



        if st.button("Generar pronóstico de PT"):

          

            resultados_pt = crear_pronosticos_generico(
                series_dict_pt,
                periodos_atras_pt,
                lags_pt,
                modelos_seleccionados=modelos_seleccionados
            )
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
# PESTAÃA MATERIA PRIMA
# ----------------------------
elif seccion == "Pronósticos MP":
    st.subheader("Materia prima desde archivo de demanda consolidado")
    st.caption("La BOM se toma de la hoja 'Products & Materials' del Excel cargado en Pronósticos PT.")



    if st.button("Ejecutar explosión de materiales"):
        try:
            if "df" not in st.session_state:
                st.warning("Primero debes generar el pronóstico de Producto Terminado.")
                st.stop()
            if "df_bom_vertical" not in st.session_state:
                st.warning("No se encontró BOM en session_state. Vuelve a cargar el Excel en Pronósticos PT.")
                st.stop()



            df = st.session_state["df"]
            df_bom_vertical = st.session_state["df_bom_vertical"]
            df_consumo = explosionar_mp(df, df_bom_vertical)
            st.session_state["df_consumo"] = df_consumo
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
                resultados_mp = crear_pronosticos_generico(
                    series_dict_mp,
                    periodos_atras_mp,
                    lags_mp,
                    modelos_seleccionados=modelos_seleccionados
                )
                df_resumen_mp = generar_resumen_mp(resultados_mp)



                st.session_state['resultados_mp'] = resultados_mp
                st.session_state['df_resumen_mp'] = df_resumen_mp



                colores_mp = generar_colores_mp(df_bom_vertical['MATERIA_PRIMA'].unique())
                st.session_state['colores_mp'] = colores_mp



                fig = graficar_pronosticos_mp(df_consumo, resultados_mp, colores_mp, lags=lags_mp)
                st.session_state['fig_mp'] = fig
            except Exception as e:
                st.error(f"Error durante el pronÃ³stico: {e}")
    # Mostrar resultados si ya existen
    if 'df_resumen_mp' in st.session_state:
        st.subheader("Resumen del pronóstico MP")
        st.dataframe(st.session_state['df_resumen_mp'], use_container_width=True)



        # Reconstruir grÃ¡fica si no estÃ¡ en session_state
        if 'fig_mp' not in st.session_state:
            st.session_state['fig_mp'] = graficar_pronosticos_mp(
                st.session_state['df_consumo'],
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





# ----------------------------
# PESTAÑA CAPACIDAD PROCESOS PLANTA
# ----------------------------
elif seccion == "Capacidad Procesos Planta":
    st.subheader("Capacidad de Procesos de Planta")
    st.caption(
        "Calcula minutos requeridos por proceso usando el pronóstico PT y los tiempos "
        "de la hoja 'Production' del gameplay-export."
    )
    if "resultados_pt" not in st.session_state:
        st.warning("Primero genera el pronóstico de Producto Terminado en la pestaña PT.")
        st.stop()
    if "df_tiempos_prod" not in st.session_state:
        st.warning("No se encontraron tiempos de Production. Vuelve a cargar el Excel en PT.")
        st.stop()
    if "df_capacidad_proc" not in st.session_state:
        st.session_state["df_capacidad_proc"] = pd.DataFrame(columns=["PROCESO", "CAPACIDAD_MIN"])
    df_pron_pt_planta = construir_pronostico_pt_planta(st.session_state["resultados_pt"])
    df_carga_proc = calcular_carga_procesos_planta(
        df_pron_pt_planta,
        st.session_state["df_tiempos_prod"],
        st.session_state["df_capacidad_proc"]
    )
    if df_carga_proc.empty:
        st.info("No hay datos suficientes para calcular capacidad por proceso.")
    else:
        if (
            "CAPACIDAD_MIN" not in df_carga_proc.columns
            or df_carga_proc["CAPACIDAD_MIN"].isna().all()
        ):
            st.info(
                "Se cargaron los tiempos de 'Production', pero no se identificó "
                "una fila de capacidad/disponibilidad en ese archivo."
            )
        st.dataframe(df_carga_proc, use_container_width=True)
        fig_cap = graficar_capacidad_procesos(df_carga_proc)
        st.plotly_chart(fig_cap, use_container_width=True)

