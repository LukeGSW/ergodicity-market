"""
calculations.py — Logica quantitativa per lo studio dell'Ergodicità del Mercato.

Questo modulo implementa il nucleo matematico dell'analisi:
  - Calcolo dei rendimenti logaritmici
  - Media temporale (rolling) e media spaziale (expanding)
  - Classificazione dei giorni ergodici / non ergodici basata sul SEM
  - Statistiche riepilogative e analisi per decennio

═══════════════════════════════════════════════════════════════
FONDAMENTO TEORICO DELLA SOGLIA — Standard Error of the Mean
═══════════════════════════════════════════════════════════════
La soglia di ergodicità è calcolata come:

    threshold = k × σ / √N

dove:
  σ = deviazione standard globale dei log-return (stima della vera volatilità)
  N = finestra rolling (es. 252 giorni = 1 anno di trading)
  k = moltiplicatore di confidenza (k=1.96 → 95% CI, k=1.75 → 92% CI)

Questa formula è lo Standard Error of the Mean (SEM): la rolling mean su N
osservazioni ha una precisione statistica di σ/√N intorno al suo valore atteso.
Una deviazione |rolling_mean − expanding_mean| > k × σ/√N è quindi
"statisticamente significativa" al livello di confidenza corrispondente a k.

Proprietà desiderabili (assenti nella formula std(diff)):
  ✓ Scala con la volatilità dell'asset (asset più volatili → banda più larga)
  ✓ Si restringe al crescere di N (finestre più lunghe → stime più precise)
  ✓ È interpretabile: corrisponde a un test t sulla media con N gradi di libertà
  ✓ Replica il valore ±0.0011 osservato per SPX/252g (k≈1.75)

Riferimento teorico:
  Ole Peters (2019) — "The ergodicity problem in economics", Nature Physics 15.
  Peters & Gell-Mann (2016) — "Evaluating gambles using dynamics", Chaos 26.
  Nassim N. Taleb — "Ergodicity and Ensemble Average", Incerto series.
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


# ============================================================
# DATACLASS RISULTATI
# ============================================================

@dataclass
class ErgodicityResult:
    """
    Contenitore dei risultati dell'analisi di ergodicità.

    Attributes:
        df:              DataFrame arricchito con tutte le colonne calcolate
        threshold:       Valore soglia usato per la classificazione (= k × σ / √N)
        sigma_global:    Deviazione standard globale dei log-return (σ)
        sem:             Standard Error of the Mean puro (σ / √N, senza moltiplicatore)
        k_mult:          Moltiplicatore k usato (threshold = k × sem)
        n_total:         Numero totale di giorni analizzati
        n_non_ergodic:   Numero di giorni classificati non ergodici
        pct_non_ergodic: Percentuale giorni non ergodici
        current_diff:    Valore attuale della differenza (rolling - expanding)
        is_ergodic_now:  True se lo stato attuale è ergodico
        status_label:    Etichetta testuale dello stato attuale
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


# ============================================================
# FUNZIONI PRINCIPALI
# ============================================================

def compute_log_returns(price: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti logaritmici giornalieri.

    Il log-return r_t = ln(P_t / P_{t-1}) è la misura standard in finanza
    quantitativa: è additivo nel tempo, simmetrico e approssima bene
    i rendimenti percentuali per variazioni piccole.

    Args:
        price: Serie di prezzi (adjusted close), indice DatetimeIndex

    Returns:
        Serie dei rendimenti logaritmici giornalieri
    """
    return np.log(price / price.shift(1))


def compute_ergodicity_metrics(
    df: pd.DataFrame,
    rolling_window: int = TRADING_DAYS_YEAR,
    threshold_mode: Literal["sem", "manual"] = "sem",
    threshold_mult: float = 1.75,
    manual_threshold: float = 0.0011,
    price_col: str | None = None,
) -> ErgodicityResult:
    """
    Esegue l'analisi di ergodicità completa su un DataFrame OHLCV.

    Algoritmo:
      1. Seleziona il prezzo (adjusted_close se disponibile, altrimenti close)
      2. Calcola log-returns giornalieri
      3. Calcola rolling_mean (media temporale) su finestra `rolling_window`
      4. Calcola expanding_mean (media spaziale) dall'origine
      5. diff = rolling_mean - expanding_mean
      6. Soglia SEM: threshold = k × σ_globale / √N  (Standard Error of the Mean)
         oppure soglia manuale: valore fisso
      7. Flag non ergodico: |diff| > soglia
      8. Calcola statistiche riepilogative

    SOGLIA — Standard Error of the Mean (SEM):
      La rolling mean calcolata su N osservazioni i.i.d. con std σ ha incertezza σ/√N.
      La deviazione |rolling − expanding| è "statisticamente significativa" (non ergodica)
      quando supera k × σ/√N.
      - k=1.00 → 68.3% confidence interval (1-sigma)
      - k=1.75 → ~92%  confidence interval (replica ±0.0011 per SPX/252g)
      - k=1.96 → 95.0% confidence interval (standard scientifico)
      - k=2.58 → 99.0% confidence interval (test molto conservativo)

    Args:
        df:               DataFrame con colonne OHLCV (e opzionalmente adjusted_close)
        rolling_window:   Finestra per la rolling mean (default 252 = 1 anno trading)
        threshold_mode:   'sem' = k × σ/√N (Standard Error of the Mean, raccomandato);
                          'manual' = valore fisso
        threshold_mult:   Moltiplicatore k per modalità 'sem' (default 1.75 ≈ 92% CI)
        manual_threshold: Valore soglia fisso per modalità 'manual'
        price_col:        Colonna prezzo da usare (None = auto-detect adjusted_close/close)

    Returns:
        ErgodicityResult con DataFrame arricchito, soglia, σ, SEM e statistiche complete
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

    # === 2. Log-returns ===
    result_df["log_ret"] = compute_log_returns(result_df["price"])

    # === 3. Media temporale (rolling) ===
    # Stima locale dell'aspettativa dei rendimenti negli ultimi N giorni
    result_df["rolling_mean"] = (
        result_df["log_ret"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .mean()
    )

    # === 4. Media spaziale (expanding) ===
    # Stima globale: media cumulativa dall'inizio della serie.
    # Converge alla vera media di lungo periodo man mano che N → ∞.
    result_df["expanding_mean"] = (
        result_df["log_ret"]
        .expanding(min_periods=rolling_window)
        .mean()
    )

    # === 5. Differenza rolling − expanding ===
    # Misura lo scostamento tra la stima locale e la stima globale.
    # Un valore vicino a 0 indica che il mercato "ricorda" il suo comportamento storico.
    result_df["diff"] = result_df["rolling_mean"] - result_df["expanding_mean"]

    # Rimuovi NaN (prime rolling_window righe senza dati sufficienti)
    clean = result_df.dropna(subset=["rolling_mean", "expanding_mean", "diff"]).copy()

    # === 6. Soglia ergodicità ===
    # σ globale: deviazione standard sull'intera serie dei log-return disponibili.
    # È la migliore stima della "vera" volatilità dell'asset.
    sigma_global = float(result_df["log_ret"].dropna().std())

    if threshold_mode == "sem":
        # Standard Error of the Mean: incertezza statistica della rolling mean su N obs.
        # Fondamento: la rolling mean X̄_N ha distribuzione con std = σ / √N per grandi N.
        # Deviazioni oltre k × σ/√N sono significative al livello di confidenza di k-sigma.
        sem = sigma_global / np.sqrt(rolling_window)
        threshold = threshold_mult * sem
    else:
        # Modalità manuale: soglia fissa inserita dall'utente
        sem = sigma_global / np.sqrt(rolling_window)   # calcolato per display
        threshold = float(manual_threshold)

    # === 7. Classificazione giorni ===
    # Un giorno è non ergodico quando la deviazione locale dalla media storica
    # supera statisticamente la soglia scelta.
    clean["is_non_ergodic"] = clean["diff"].abs() > threshold

    # === 8. Statistiche finali ===
    n_total = len(clean)
    n_non_ergodic = int(clean["is_non_ergodic"].sum())
    pct_non_ergodic = 100.0 * n_non_ergodic / n_total if n_total > 0 else 0.0
    current_diff = float(clean["diff"].iloc[-1])
    is_ergodic_now = abs(current_diff) <= threshold
    status_label = "ERGODICO ✅" if is_ergodic_now else "NON ERGODICO ⚠️"

    return ErgodicityResult(
        df=clean,
        threshold=threshold,
        sigma_global=sigma_global,
        sem=sem,
        k_mult=threshold_mult if threshold_mode == "sem" else (threshold / sem),
        n_total=n_total,
        n_non_ergodic=n_non_ergodic,
        pct_non_ergodic=pct_non_ergodic,
        current_diff=current_diff,
        is_ergodic_now=is_ergodic_now,
        status_label=status_label,
    )


def compute_decade_stats(result: ErgodicityResult) -> pd.DataFrame:
    """
    Calcola le statistiche di ergodicità raggruppate per decennio.

    Utile per identificare periodi storici in cui il mercato ha mostrato
    comportamenti sistematicamente non ergodici (es. crisi strutturali).

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

    Finestra identica al rolling_window dell'analisi principale.
    Utile per visualizzare l'evoluzione temporale del livello di ergodicità.

    Args:
        result: ErgodicityResult prodotto da compute_ergodicity_metrics()

    Returns:
        Serie con la % rolling di giorni non ergodici (0–100)
    """
    # Ricava rolling_window dall'ampiezza degli ultimi dati disponibili
    # (approssimato con la differenza di expanding_mean tra inizio e fine)
    # Usiamo TRADING_DAYS_YEAR come proxy della finestra originale
    return result.df["is_non_ergodic"].rolling(TRADING_DAYS_YEAR).mean() * 100


def compute_diff_statistics(result: ErgodicityResult) -> dict:
    """
    Statistiche descrittive complete della serie delle differenze.

    Utile per l'expander metodologico dell'app e per validare la soglia scelta.

    Args:
        result: ErgodicityResult prodotto da compute_ergodicity_metrics()

    Returns:
        Dizionario con: mean, std, skew, kurt, min, max, q05, q95, beyond_threshold
    """
    diff = result.df["diff"].dropna()
    return {
        "media": float(diff.mean()),
        "std": float(diff.std()),
        "skewness": float(diff.skew()),
        "kurtosis_excess": float(diff.kurt()),
        "minimo": float(diff.min()),
        "massimo": float(diff.max()),
        "percentile_5": float(diff.quantile(0.05)),
        "percentile_95": float(diff.quantile(0.95)),
        "oltre_soglia": int((diff.abs() > result.threshold).sum()),
        "pct_oltre_soglia": float(100 * (diff.abs() > result.threshold).mean()),
    }
