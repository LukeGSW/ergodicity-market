"""
calculations.py — Logica quantitativa per lo studio dell'Ergodicità del Mercato.

Questo modulo implementa il nucleo matematico dell'analisi. Sono disponibili
DUE metodi di rilevazione della non-ergodicità:

  • method="vol_drag"  (DEFAULT, raccomandato per individuare ECCESSI)
      Confronta la media d'insieme (aritmetica, sui rendimenti semplici) con la
      media temporale (geometrica, sui log-return), entrambe sulla STESSA finestra
      rolling. La loro differenza è il *volatility drag* ≈ ½·σ²_locale: è la vera
      misura di non-ergodicità per dinamiche moltiplicative (Peters). L'eccesso è
      rilevato quando il drag supera di k deviazioni la sua norma adattiva (EWMA).

  • method="drift_divergence"  (LEGACY, comportamento storico dell'app)
      Confronta la rolling mean dei log-return con la loro media expanding
      dall'origine; soglia = Standard Error of the Mean (k·σ/√N).

═══════════════════════════════════════════════════════════════
PERCHÉ vol_drag INDIVIDUA GLI ECCESSI E drift_divergence NO
═══════════════════════════════════════════════════════════════
Nel metodo legacy entrambe le medie confrontate (rolling ed expanding) sono
medie TEMPORALI della stessa traiettoria: misurano se il drift recente differisce
da quello storico (cambio di regime), non l'ergodicità. Inoltre l'expanding mean
si "congela" (1/N di sensibilità dopo migliaia di osservazioni) → la gamba
spaziale diventa una costante inerte. Il segnale è una media di rendimenti: un
crollo di −10% in un giorno sposta la media a 252g di appena ≈ −0.0004. È quindi
un filtro di TREND, cieco agli eccessi.

Il metodo vol_drag risolve entrambi i problemi:

    g_insieme  = mean_N(rendimenti semplici)      (aritmetica ≈ valore atteso)
    g_tempo    = expm1(mean_N(log-return))        (geometrica ≈ ciò che compone)
    ne_gap     = g_insieme − g_tempo  ≈  ½·σ²_locale   (≥ 0, il volatility drag)

ne_gap è guidato dalla VARIANZA locale ed esplode esattamente quando la
volatilità spara — cioè durante gli eccessi non ergodici. La soglia usa una
volatilità LOCALE (non un σ globale costante) e il segnale è standardizzato in
z-score contro la propria norma adattiva (EWMA), così "z>k" ha lo stesso
significato in ogni regime e su asset diversi.

Riferimenti teorici:
  Ole Peters (2019) — "The ergodicity problem in economics", Nature Physics 15.
  Peters & Gell-Mann (2016) — "Evaluating gambles using dynamics", Chaos 26.
  Nassim N. Taleb — "Skin in the Game" / Incerto series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


# ============================================================
# COSTANTI
# ============================================================

TRADING_DAYS_YEAR = 252   # giorni di trading in un anno solare
MIN_PERIODS_FACTOR = 1    # min_periods = ROLLING_WINDOW * fattore

Method = Literal["vol_drag", "drift_divergence"]


# ============================================================
# DATACLASS RISULTATI
# ============================================================

@dataclass
class ErgodicityResult:
    """
    Contenitore dei risultati dell'analisi di ergodicità.

    Campi comuni:
        df:              DataFrame arricchito con tutte le colonne calcolate
        threshold:       Soglia di classificazione. In 'drift_divergence' è
                         k·σ/√N (unità di Δ); in 'vol_drag' è il cutoff k sullo
                         z-score (unità di deviazioni standard).
        sigma_global:    Deviazione standard globale dei log-return (σ)
        sem:             Standard Error of the Mean (σ / √N), per display
        k_mult:          Moltiplicatore/cutoff k usato
        n_total:         Numero totale di giorni analizzati
        n_non_ergodic:   Numero di giorni classificati non ergodici
        pct_non_ergodic: Percentuale giorni non ergodici
        current_diff:    Valore attuale del segnale grezzo (Δ oppure ne_gap)
        is_ergodic_now:  True se lo stato attuale è ergodico
        status_label:    Etichetta testuale dello stato attuale

    Campi aggiuntivi:
        method:          'vol_drag' | 'drift_divergence'
        rolling_window:  Finestra N usata (giorni)
        current_z:       z-score attuale (NaN in modalità legacy)
    """
    df: pd.DataFrame
    threshold: float
    sigma_global: float
    sem: float
    k_mult: float
    n_total: int
    n_non_ergodic: int
    pct_non_ergodic: float
    current_diff: float
    is_ergodic_now: bool
    status_label: str
    method: Method = "vol_drag"
    rolling_window: int = TRADING_DAYS_YEAR
    current_z: float = float("nan")


# ============================================================
# FUNZIONI PRINCIPALI
# ============================================================

def compute_log_returns(price: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti logaritmici giornalieri.

    Il log-return r_t = ln(P_t / P_{t-1}) è additivo nel tempo e rappresenta
    la media TEMPORALE (geometrica) della crescita: è ciò che una singola
    traiettoria effettivamente compone.

    Args:
        price: Serie di prezzi (adjusted close), indice DatetimeIndex

    Returns:
        Serie dei rendimenti logaritmici giornalieri
    """
    return np.log(price / price.shift(1))


def compute_simple_returns(price: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti semplici giornalieri r_t = P_t / P_{t-1} − 1.

    La media aritmetica dei rendimenti semplici è la stima della media
    D'INSIEME (ensemble / valore atteso): quanto renderebbe in media questo
    tipo di asset su un'unica realizzazione futura.

    Args:
        price: Serie di prezzi (adjusted close), indice DatetimeIndex

    Returns:
        Serie dei rendimenti semplici giornalieri
    """
    return price / price.shift(1) - 1.0


def compute_ergodicity_metrics(
    df: pd.DataFrame,
    rolling_window: int = TRADING_DAYS_YEAR,
    method: Method = "vol_drag",
    threshold_mode: Literal["sem", "manual"] = "sem",
    threshold_mult: float = 1.75,
    manual_threshold: float = 0.0011,
    vol_baseline_halflife: int | None = None,
    price_col: str | None = None,
) -> ErgodicityResult:
    """
    Esegue l'analisi di ergodicità completa su un DataFrame OHLCV.

    Args:
        df:               DataFrame con colonne OHLCV (e opzionalmente adjusted_close)
        rolling_window:   Finestra N per le medie rolling (default 252 = 1 anno)
        method:           'vol_drag' (default) o 'drift_divergence' (legacy)
        threshold_mode:   'sem' = k·σ/√N, 'manual' = valore fisso
                          (rilevante solo per 'drift_divergence')
        threshold_mult:   Moltiplicatore k. In 'vol_drag' è il cutoff z (one-sided)
        manual_threshold: Soglia fissa per 'drift_divergence' + 'manual'
        vol_baseline_halflife: Halflife EWMA per la norma del drag (default = max(2N, 252))
        price_col:        Colonna prezzo (None = auto-detect adjusted_close/close)

    Returns:
        ErgodicityResult con DataFrame arricchito e statistiche complete.
    """
    result_df = df.copy()

    # === 1. Selezione prezzo ===
    if price_col:
        result_df["price"] = result_df[price_col].astype(float)
    elif "adjusted_close" in result_df.columns:
        result_df["price"] = result_df["adjusted_close"].astype(float)
    elif "close" in result_df.columns:
        result_df["price"] = result_df["close"].astype(float)
    else:
        raise ValueError("DataFrame non contiene colonne 'adjusted_close' o 'close'.")

    # === 2. Rendimenti ===
    result_df["log_ret"] = compute_log_returns(result_df["price"])
    result_df["simple_ret"] = compute_simple_returns(result_df["price"])

    # === 3. Media temporale (rolling, geometrica) ===
    # Stima locale della crescita realmente composta dalla traiettoria.
    result_df["rolling_mean"] = (
        result_df["log_ret"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .mean()
    )

    # === 4. Volatilità locale (condizionale) ===
    # Sostituisce il σ globale costante: la banda si adatta al regime corrente.
    result_df["sigma_local"] = (
        result_df["log_ret"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .std()
    )

    # σ globale: mantenuto per display e per la modalità legacy.
    sigma_global = float(result_df["log_ret"].dropna().std())
    sem = sigma_global / np.sqrt(rolling_window)

    if method == "vol_drag":
        result_df, threshold, k_used, dropna_cols = _compute_vol_drag(
            result_df, rolling_window, threshold_mult, vol_baseline_halflife
        )
    else:
        result_df, threshold, k_used, dropna_cols = _compute_drift_divergence(
            result_df, rolling_window, sigma_global, sem,
            threshold_mode, threshold_mult, manual_threshold,
        )

    # === Pulizia NaN (righe senza dati sufficienti) ===
    clean = result_df.dropna(subset=dropna_cols).copy()

    # === Statistiche finali ===
    n_total = len(clean)
    n_non_ergodic = int(clean["is_non_ergodic"].sum())
    pct_non_ergodic = 100.0 * n_non_ergodic / n_total if n_total > 0 else 0.0
    current_diff = float(clean["diff"].iloc[-1]) if n_total else float("nan")

    if method == "vol_drag":
        current_z = float(clean["z_score"].iloc[-1]) if n_total else float("nan")
        is_ergodic_now = (current_z <= threshold) if n_total else True
    else:
        current_z = float("nan")
        is_ergodic_now = (abs(current_diff) <= threshold) if n_total else True

    status_label = "ERGODICO ✅" if is_ergodic_now else "NON ERGODICO ⚠️"

    return ErgodicityResult(
        df=clean,
        threshold=threshold,
        sigma_global=sigma_global,
        sem=sem,
        k_mult=k_used,
        n_total=n_total,
        n_non_ergodic=n_non_ergodic,
        pct_non_ergodic=pct_non_ergodic,
        current_diff=current_diff,
        is_ergodic_now=is_ergodic_now,
        status_label=status_label,
        method=method,
        rolling_window=rolling_window,
        current_z=current_z,
    )


# ============================================================
# IMPLEMENTAZIONE DEI DUE METODI
# ============================================================

def _compute_vol_drag(
    df: pd.DataFrame,
    rolling_window: int,
    threshold_mult: float,
    vol_baseline_halflife: int | None,
) -> tuple[pd.DataFrame, float, float, list[str]]:
    """
    Metodo vol_drag: gap aritmetico−geometrico standardizzato.

    ne_gap(t) = mean_N(simple_ret) − expm1(mean_N(log_ret))  ≈ ½·σ²_locale
    z(t)      = (ne_gap − EWMA_mean(ne_gap)) / EWMA_std(ne_gap)
    eccesso   ⟺ z(t) > k   (one-sided: solo un drag anomalmente alto è un eccesso)

    Per compatibilità con grafici/export:
      • 'rolling_mean'   resta la media temporale (geometrica, log)
      • 'expanding_mean' = media d'insieme adattiva (aritmetica) — NON più inerte
      • 'diff'           = ne_gap (il volatility drag)
    """
    # Media d'insieme (aritmetica) sulla stessa finestra N
    df["ensemble_mean"] = (
        df["simple_ret"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .mean()
    )
    # Media temporale in unità di rendimento semplice per-periodo (geometrica)
    g_time_simple = np.expm1(df["rolling_mean"])

    # Volatility drag ≈ ½·σ²_locale (≥ 0)
    df["ne_gap"] = df["ensemble_mean"] - g_time_simple

    # Alias per compatibilità: la "media spaziale" ora è l'ensemble adattivo
    df["expanding_mean"] = df["ensemble_mean"]
    df["diff"] = df["ne_gap"]

    # Standardizzazione contro una norma ADATTIVA (EWMA), non un σ globale fisso.
    # La baseline è volutamente PIÙ LENTA della finestra N (≈ norma pluriennale):
    # se fosse veloce come N, assorbirebbe i burst e li smorzerebbe. Default = max(2N, 252).
    hl = vol_baseline_halflife or max(2 * rolling_window, 252)
    base = df["ne_gap"].ewm(halflife=hl, min_periods=rolling_window).mean()
    scale = df["ne_gap"].ewm(halflife=hl, min_periods=rolling_window).std()
    scale = scale.replace(0.0, np.nan)
    df["z_score"] = (df["ne_gap"] - base) / scale

    threshold = float(threshold_mult)        # cutoff z, one-sided
    df["is_non_ergodic"] = df["z_score"] > threshold

    dropna_cols = ["rolling_mean", "expanding_mean", "diff", "z_score"]
    return df, threshold, float(threshold_mult), dropna_cols


def _compute_drift_divergence(
    df: pd.DataFrame,
    rolling_window: int,
    sigma_global: float,
    sem: float,
    threshold_mode: str,
    threshold_mult: float,
    manual_threshold: float,
) -> tuple[pd.DataFrame, float, float, list[str]]:
    """
    Metodo legacy: rolling mean vs expanding mean dei log-return, soglia SEM.
    Comportamento identico alla versione storica (riproducibilità).
    """
    # Media spaziale (expanding) dall'origine
    df["expanding_mean"] = (
        df["log_ret"]
        .expanding(min_periods=rolling_window)
        .mean()
    )
    df["ensemble_mean"] = np.nan          # non usato in questo metodo
    df["ne_gap"] = np.nan
    df["z_score"] = np.nan

    df["diff"] = df["rolling_mean"] - df["expanding_mean"]

    if threshold_mode == "sem":
        threshold = threshold_mult * sem
        k_used = threshold_mult
    else:
        threshold = float(manual_threshold)
        k_used = threshold / sem if sem else float("nan")

    df["is_non_ergodic"] = df["diff"].abs() > threshold

    dropna_cols = ["rolling_mean", "expanding_mean", "diff"]
    return df, threshold, k_used, dropna_cols


def compute_decade_stats(result: ErgodicityResult) -> pd.DataFrame:
    """
    Calcola le statistiche di ergodicità raggruppate per decennio.

    Args:
        result: ErgodicityResult prodotto da compute_ergodicity_metrics()

    Returns:
        DataFrame con una riga per decennio e colonne statistiche
    """
    df = result.df.copy()
    df["decade"] = (df.index.year // 10) * 10

    stats = (
        df.groupby("decade")
        .agg(
            giorni_totali=("log_ret", "count"),
            giorni_non_ergodici=("is_non_ergodic", "sum"),
            pct_non_ergodici=("is_non_ergodic", lambda x: 100.0 * x.mean()),
            rolling_mean_medio=("rolling_mean", "mean"),
            expanding_mean_medio=("expanding_mean", "mean"),
            diff_medio=("diff", "mean"),
            diff_std=("diff", "std"),
        )
        .reset_index()
    )
    stats["decade"] = stats["decade"].astype(str) + "s"
    return stats


def compute_rolling_pct_non_ergodic(result: ErgodicityResult) -> pd.Series:
    """
    Calcola la percentuale rolling di giorni non ergodici.

    Usa la STESSA finestra dell'analisi principale (result.rolling_window),
    non più un valore fisso a 252.

    Args:
        result: ErgodicityResult prodotto da compute_ergodicity_metrics()

    Returns:
        Serie con la % rolling di giorni non ergodici (0–100)
    """
    return result.df["is_non_ergodic"].rolling(result.rolling_window).mean() * 100


def compute_diff_statistics(result: ErgodicityResult) -> dict:
    """
    Statistiche descrittive del segnale di rilevazione.

    In 'vol_drag' il segnale descritto è lo z-score (ciò che l'istogramma mostra);
    in 'drift_divergence' è la differenza Δ. I conteggi "oltre soglia" usano
    sempre il flag effettivo is_non_ergodic, così sono coerenti in entrambi i modi.

    Args:
        result: ErgodicityResult prodotto da compute_ergodicity_metrics()

    Returns:
        Dizionario con: media, std, skewness, kurtosis_excess, min, max,
        percentili e conteggi oltre soglia.
    """
    signal_col = "z_score" if result.method == "vol_drag" else "diff"
    sig = result.df[signal_col].dropna()
    n_flag = int(result.df["is_non_ergodic"].sum())
    n = len(result.df)
    return {
        "media": float(sig.mean()),
        "std": float(sig.std()),
        "skewness": float(sig.skew()),
        "kurtosis_excess": float(sig.kurt()),
        "minimo": float(sig.min()),
        "massimo": float(sig.max()),
        "percentile_5": float(sig.quantile(0.05)),
        "percentile_95": float(sig.quantile(0.95)),
        "oltre_soglia": n_flag,
        "pct_oltre_soglia": float(100.0 * n_flag / n) if n else 0.0,
    }
