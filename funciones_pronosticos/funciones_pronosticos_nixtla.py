# ============================================================
# ✅ NUEVO (2026-01): Motor Nixtla StatsForecast (Backtesting rápido)
# ============================================================
# Este bloque agrega funciones reutilizables para:
# - Transformar data Mototrak (Turn/REGIONAL/PRODUCTO o MATERIA_PRIMA) a formato StatsForecast (unique_id/ds/y)
# - Ejecutar cross_validation (backtesting) automático
# - Calcular Score% = MAE% + |Sesgo%|
# - Seleccionar el mejor modelo por serie y obtener pronóstico final filtrado
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        HoltWinters,
        SimpleExponentialSmoothingOptimized,
        WindowAverage,
        Naive,
        SeasonalNaive,
        SeasonalWindowAverage,
        ADIDA,
        CrostonOptimized,
        MSTL as SF_MSTL,
        MFLES,
        AutoETS,
        AutoTheta,
        HistoricAverage,
    )
    # Modelo de regresión vía sklearn (opcional)
    try:
        from statsforecast.models import SklearnModel
        from sklearn.linear_model import LinearRegression as SkLinearRegression
        _SKLEARN_OK = True
    except Exception:
        SklearnModel = None
        SkLinearRegression = None
        _SKLEARN_OK = False

    _STATSFORECAST_OK = True
except Exception:
    StatsForecast = None
    _STATSFORECAST_OK = False


def _turn_to_ds(
    turn: pd.Series,
    start_date: str = "2020-01-06",
    freq_days: int = 7
) -> pd.Series:
    """
    Convierte Turn (entero) a fechas (ds) usando un origen fijo.
    - start_date: lunes base
    - freq_days: 7 para semanal
    """
    t0 = int(pd.Series(turn).min())
    base = pd.to_datetime(start_date)
    return base + pd.to_timedelta((turn.astype(int) - t0) * freq_days, unit="D")


def mototrak_to_statsforecast_df(
    df: pd.DataFrame,
    col_id_1: str,
    col_id_2: str,
    col_y: str = "DEMANDA",
    col_turn: str = "Turn",
    start_date: str = "2020-01-06",
    freq: str = "W-MON",
) -> Tuple[pd.DataFrame, Dict[str, int], str]:
    """
    Convierte un df Mototrak a formato largo para StatsForecast:
      unique_id, ds, y

    Retorna:
      - df_ts (unique_id, ds, y)
      - mapping_turn0: dict {unique_id: turn_min} para reconvertir ds->Turn si se requiere
      - freq: frecuencia usada
    """
    df2 = df[[col_turn, col_id_1, col_id_2, col_y]].copy()
    df2 = df2.dropna(subset=[col_y])
    df2[col_turn] = df2[col_turn].astype(int)

    # unique_id (ej: "NORTE|MOTO" o "CENTRO|MP_1")
    df2["unique_id"] = df2[col_id_1].astype(str) + "|" + df2[col_id_2].astype(str)

    # ds (fecha ficticia semanal) por serie
    mapping_turn0 = df2.groupby("unique_id")[col_turn].min().to_dict()
    df2["ds"] = df2.groupby("unique_id")[col_turn].transform(lambda s: _turn_to_ds(s, start_date=start_date))

    df_ts = df2.sort_values(["unique_id", "ds"]).rename(columns={col_y: "y"})[["unique_id", "ds", "y"]].reset_index(drop=True)

    return df_ts, mapping_turn0, freq


def calcular_metricas_basicas_sf(y: pd.Series, pred: pd.Series) -> Optional[Dict[str, float]]:
    """
    Métricas globales (estilo tu app):
      mae_porc = sum(|y - pred|) / sum(y)
      sesgo_porc = sum(y - pred) / sum(y)
      score_porc = mae_porc + |sesgo_porc|
    """
    dfm = pd.DataFrame({"y": y, "pred": pred}).dropna()
    suma_y = dfm["y"].sum()
    if suma_y == 0 or len(dfm) == 0:
        return None
    err = dfm["y"] - dfm["pred"]
    mae_porc = err.abs().sum() / suma_y
    sesgo_porc = err.sum() / suma_y
    score_porc = mae_porc + abs(sesgo_porc)
    return {"mae_porc": float(mae_porc), "sesgo_porc": float(sesgo_porc), "score_porc": float(score_porc)}


def evaluar_score_statsforecast(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el output de sf.cross_validation y calcula métricas por modelo.
    Devuelve un df ordenado por score_porc.
    """
    columnas_fijas = ["unique_id", "ds", "cutoff", "y"]
    modelos = [c for c in cv_df.columns if c not in columnas_fijas]
    resultados = {}
    for m in modelos:
        met = calcular_metricas_basicas_sf(cv_df["y"], cv_df[m])
        resultados[m] = met
    df_res = (
        pd.DataFrame.from_dict(resultados, orient="index")
        .reset_index()
        .rename(columns={"index": "modelo"})
        .sort_values("score_porc")
        .reset_index(drop=True)
    )
    return df_res


def seleccionar_mejor_modelo_por_serie_sf(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada unique_id selecciona el modelo con menor score_porc.
    """
    columnas_fijas = ["unique_id", "ds", "cutoff", "y"]
    modelos = [c for c in cv_df.columns if c not in columnas_fijas]
    out = []
    for uid in cv_df["unique_id"].unique():
        df_uid = cv_df[cv_df["unique_id"] == uid]
        mejor_modelo = None
        mejor_score = float("inf")
        for m in modelos:
            met = calcular_metricas_basicas_sf(df_uid["y"], df_uid[m])
            if met is None:
                continue
            if met["score_porc"] < mejor_score:
                mejor_score = met["score_porc"]
                mejor_modelo = m
        out.append({"unique_id": uid, "modelo": mejor_modelo})
    return pd.DataFrame(out)


def filtrar_por_mejores_modelos_sf(df: pd.DataFrame, mejores_modelos: pd.DataFrame) -> pd.DataFrame:
    """
    Pasa df a formato largo y deja solo las predicciones del mejor modelo por unique_id.
    Funciona tanto para cv_df (tiene cutoff,y) como para forecasts (no tiene cutoff,y).
    """
    posibles_id = ["unique_id", "ds", "cutoff", "y"]
    id_vars = [c for c in posibles_id if c in df.columns]
    modelos = [c for c in df.columns if c not in id_vars]
    df_largo = (
        df.melt(id_vars=id_vars, value_vars=modelos, var_name="modelo", value_name="pred")
        .dropna(subset=["pred"])
    )
    return df_largo.merge(mejores_modelos, on=["unique_id", "modelo"])


def score_global_mejores_sf(df_mejores_filtrado: pd.DataFrame) -> pd.DataFrame:
    met = calcular_metricas_basicas_sf(df_mejores_filtrado["y"], df_mejores_filtrado["pred"])
    if met is None:
        return pd.DataFrame([{"mae_porc": np.nan, "sesgo_porc": np.nan, "score_porc": np.nan}])
    return pd.DataFrame([met])


def construir_modelos_statsforecast(
    usar_hw: bool = True,
    usar_mstl: bool = True,
    usar_ses: bool = False,
    pms_windows: Optional[List[int]] = None,
    incluir_baselines: bool = True,
    season_length: int = 52,
    incluir_regresion_lineal: bool = True,
    modelos_seleccionados=None,
) -> List:
    """
    Crea una lista de modelos StatsForecast (sin ARIMAs).
    Si modelos_seleccionados viene desde la UI, filtra qué modelos se construyen.
    """

    if not _STATSFORECAST_OK:
        raise ImportError("StatsForecast no está instalado. Agrega statsforecast a requirements.txt")

    # Defaults
    pms_windows = pms_windows or [3, 6, 12]

    # ----------------------------
    # Filtro por selección de modelos (UI)
    # ----------------------------
    if modelos_seleccionados:
        modelos_set = set(modelos_seleccionados)

        # Promedios móviles seleccionados en UI: pm_3, pm_6, pm_12
        pms_windows = [w for w in pms_windows if f"pm_{w}" in modelos_set]

        # Flags principales
        usar_hw = ("hw" in modelos_set) or ("holt_winters" in modelos_set)
        usar_mstl = ("mstl" in modelos_set)
        usar_ses = ("ses_opt" in modelos_set)  # si algún día lo expones en la UI

        # Baselines: solo si al menos uno fue pedido
        baselines_set = {"naive", "snaive", "hist_avg", "autoets", "autotheta", "croston", "adida", "mfles"}
        incluir_baselines = len(modelos_set.intersection(baselines_set)) > 0

        incluir_regresion_lineal = incluir_regresion_lineal and ("linear_reg" in modelos_set)

    # ----------------------------
    # Construcción de modelos
    # ----------------------------
    modelos = []

    # Holt-Winters
    if usar_hw:
        modelos.append(HoltWinters(season_length=season_length, alias="hw"))

    # MSTL
    if usar_mstl:
        modelos.append(SF_MSTL(season_length=season_length, alias=f"mstl_{season_length}"))

    # SES
    if usar_ses:
        modelos.append(SimpleExponentialSmoothingOptimized(alias="ses_opt"))

    # Promedios móviles (WindowAverage)
    for w in pms_windows:
        modelos.append(WindowAverage(window_size=w, alias=f"pms_{w}"))

    # Baselines (pero ahora filtrados si llegó selección)
    if incluir_baselines:
        # OJO: si modelos_seleccionados existe, solo agregamos lo que pidieron
        # Si no existe, agregamos todos como antes (comportamiento original)
        if modelos_seleccionados:
            if "naive" in modelos_set:
                modelos.append(Naive(alias="naive"))
            if "hist_avg" in modelos_set:
                modelos.append(HistoricAverage(alias="hist_avg"))
            if "snaive" in modelos_set:
                modelos.append(SeasonalNaive(season_length=season_length, alias=f"snaive_{season_length}"))
            # opcionales
            if "croston" in modelos_set:
                modelos.append(CrostonOptimized(alias="croston"))
            if "adida" in modelos_set:
                modelos.append(ADIDA(alias="adida"))
            if "autoets" in modelos_set:
                modelos.append(AutoETS(season_length=season_length, alias="autoets"))
            if "autotheta" in modelos_set:
                modelos.append(AutoTheta(season_length=season_length, alias="autotheta"))
            if "mfles" in modelos_set:
                modelos.append(MFLES(season_length=season_length, alias="mfles"))
        else:
            # Comportamiento original (si no hay UI)
            modelos.append(Naive(alias="naive"))
            modelos.append(HistoricAverage(alias="hist_avg"))
            modelos.append(SeasonalNaive(season_length=season_length, alias=f"snaive_{season_length}"))
            modelos.append(CrostonOptimized(alias="croston"))
            modelos.append(ADIDA(alias="adida"))
            modelos.append(AutoETS(season_length=season_length, alias="autoets"))
            modelos.append(AutoTheta(season_length=season_length, alias="autotheta"))
            modelos.append(MFLES(season_length=season_length, alias="mfles"))

    # Regresión lineal (tendencia)
    if incluir_regresion_lineal and _SKLEARN_OK:
        modelos.append(SklearnModel(SkLinearRegression(), alias="linreg"))
    elif incluir_regresion_lineal and not _SKLEARN_OK:
        print("⚠️ sklearn no disponible -> se omite modelo 'linreg'.")

    return modelos


def backtesting_statsforecast_mototrak(
    df_ts: pd.DataFrame,
    modelos: List,
    h: int,
    step_size: int,
    n_windows: int,
    freq: str = "W-MON",
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, object]:
    """
    Ejecuta cross_validation de StatsForecast y retorna:
    - cv_df
    - sf (objeto StatsForecast entrenable para forecast final)
    """
    if not _STATSFORECAST_OK:
        raise ImportError("StatsForecast no está instalado.")

    sf = StatsForecast(models=modelos, freq=freq, n_jobs=n_jobs, fallback_model=Naive(alias="naive"))
    cv_df = sf.cross_validation(df=df_ts, h=h, step_size=step_size, n_windows=n_windows, refit=True)
    return cv_df, sf

def score_por_serie_mejor_modelo(cv_mejor: pd.DataFrame) -> pd.DataFrame:
    """
    cv_mejor: cross_validation filtrado solo con el modelo ganador por unique_id.
    Debe tener columnas: unique_id, y, y_hat (o el nombre de pred), y idealmente cutoff.
    """
    df = cv_mejor.copy()

    # Asegurar nombres
    if "y_hat" not in df.columns:
        # si viene con otro nombre, ajustamos a la primera columna numérica distinta de y
        posibles = [c for c in df.columns if c not in ["unique_id", "ds", "cutoff", "y"]]
        if posibles:
            df = df.rename(columns={posibles[0]: "y_hat"})

    df["err"] = df["y_hat"] - df["y"]
    df["abs_err"] = df["err"].abs()

    out = (
        df.groupby("unique_id", as_index=False)
          .agg(
              mae=("abs_err", "mean"),
              bias=("err", "mean"),
              y_mean=("y", "mean"),
          )
    )

    # métricas %
    out["mae_porc"] = (out["mae"] / out["y_mean"].replace(0, np.nan)) * 100
    out["sesgo_porc"] = (out["bias"] / out["y_mean"].replace(0, np.nan)) * 100
    out["score_porc"] = out["mae_porc"].abs() + out["sesgo_porc"].abs()

    return out[["unique_id", "mae_porc", "sesgo_porc", "score_porc"]]

def ejecutar_motor_statsforecast(
    df_original: pd.DataFrame,
    modo: str,
    h: int = 4,
    step_size: int = 1,
    n_windows: int = 8,
    season_length: int = 52,
    pms_windows: Optional[List[int]] = None,
    incluir_regresion_lineal: bool = True,
    modelos_seleccionados=None,   # 👈 NUEVO
):
    """
    Motor completo (listo para usar desde Streamlit):
    - modo = "PT" (producto terminado) o "MP" (materia prima)
    Devuelve:
      dict con cv_df, mejores_modelos, score_global, forecasts_filtrados, df_scores_modelos
    """
    if modo not in ("PT", "MP"):
        raise ValueError("modo debe ser 'PT' o 'MP'")

    if modo == "PT":
        df_ts, _, freq = mototrak_to_statsforecast_df(
            df_original,
            col_id_1="REGIONAL",
            col_id_2="PRODUCTO",
            col_y="DEMANDA",
            col_turn="Turn",
            freq="W-MON",
        )
    else:
        df_ts, _, freq = mototrak_to_statsforecast_df(
            df_original,
            col_id_1="REGIONAL",
            col_id_2="MATERIA_PRIMA",
            col_y="DEMANDA",
            col_turn="Turn",
            freq="W-MON",
        )

    modelos = construir_modelos_statsforecast(
        usar_hw=True,
        usar_mstl=True,
        usar_ses=False,
        pms_windows=pms_windows or [3, 6, 12],
        incluir_baselines=True,
        season_length=season_length,
        incluir_regresion_lineal=incluir_regresion_lineal,
        modelos_seleccionados=modelos_seleccionados,   # 👈 NUEVO
    )

    cv_df, sf = backtesting_statsforecast_mototrak(
        df_ts=df_ts,
        modelos=modelos,
        h=h,
        step_size=step_size,
        n_windows=n_windows,
        freq=freq,
        n_jobs=-1,
    )

    df_scores_modelos = evaluar_score_statsforecast(cv_df)
    mejores_modelos = seleccionar_mejor_modelo_por_serie_sf(cv_df)
    df_mejores_filtrado = filtrar_por_mejores_modelos_sf(cv_df, mejores_modelos)
    score_global = score_global_mejores_sf(df_mejores_filtrado)

    # ✅ Score por serie (necesario para hacer merge en la app)
    cv_mejor_uid = df_mejores_filtrado.rename(columns={"pred": "y_hat"})
    score_por_serie = score_por_serie_mejor_modelo(cv_mejor_uid)

    # Forecast final (h pasos)
    forecasts = sf.forecast(df=df_ts, h=h)
    forecasts_filtrados = filtrar_por_mejores_modelos_sf(forecasts, mejores_modelos)

    # Descomponer unique_id -> REGIONAL + ITEM
    forecasts_filtrados[["REGIONAL", "ITEM"]] = forecasts_filtrados["unique_id"].str.split("|", expand=True)
    if modo == "PT":
        forecasts_filtrados = forecasts_filtrados.rename(columns={"ITEM": "PRODUCTO"})
    else:
        forecasts_filtrados = forecasts_filtrados.rename(columns={"ITEM": "MATERIA_PRIMA"})

    return {
        "df_ts": df_ts,
        "cv_df": cv_df,
        "df_scores_modelos": df_scores_modelos,
        "mejores_modelos": mejores_modelos,
        "score_global": score_global,
        "score_por_serie": score_por_serie,
        "forecasts_filtrados": forecasts_filtrados,
    }
