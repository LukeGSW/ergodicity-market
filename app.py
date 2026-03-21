"""
export.py — ergodicity-market
==============================
Costruisce il JSON di esportazione dell'analisi di ergodicità, strutturato
per la ricerca di insight e alpha su dati storici.

Sezioni del JSON:
  metadata          → parametri dell'analisi e statistiche globali
  time_series       → serie completa giornaliera con tutte le feature calcolate
  non_ergodic_runs  → periodi continui di non-ergodicità (cluster)
  regime_alpha      → statistiche forward-return per regime ergodico/non-ergodico
  alpha_signals     → eventi di transizione regime pre-filtrati (entry/exit signals)
  decade_stats      → analisi aggregata per decennio
  statistical_summary → distribuzione della differenza Δ, autocorrelazione

Uso in app.py:
    from src.export import build_ergodicity_export
    payload = build_ergodicity_export(result, decade_stats, diff_stats)
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("⬇ Esporta JSON", json_bytes, "ergodicity_export.json", "application/json")
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from datetime import date

from src.calculations import ErgodicityResult


# ================================================================
# FORWARD RETURNS FORWARD UTILITY (locali al modulo)
# ================================================================

_FWD_WINDOWS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "12M": 252}


def _add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge colonne fwd_1W / fwd_1M / fwd_3M / fwd_6M / fwd_12M
    calcolate come rendimento log cumulativo futuro (in %).
    Usate solo nel modulo export, non alterano ErgodicityResult.
    """
    out = df.copy()
    for label, days in _FWD_WINDOWS.items():
        out[f"fwd_{label}"] = out["log_ret"].rolling(days).sum().shift(-days) * 100
    return out


# ================================================================
# UTILITY
# ================================================================

def _safe_float(x) -> float | None:
    """Converte in float Python nativo; None se NaN/inf."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 8)
    except (TypeError, ValueError):
        return None


def _sharpe(series: pd.Series) -> float | None:
    """Sharpe ratio semplice: mean / std (annualizzato √252)."""
    s = series.dropna()
    if len(s) < 5 or s.std() == 0:
        return None
    return _safe_float((s.mean() / s.std()) * np.sqrt(252))


def _hit_rate(series: pd.Series) -> float | None:
    """Percentuale di osservazioni con rendimento > 0."""
    s = series.dropna()
    if len(s) == 0:
        return None
    return round(float((s > 0).sum() / len(s) * 100), 2)


# ================================================================
# SEZIONE 1 — TIME SERIES
# ================================================================

def _build_time_series(df_fwd: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Restituisce la lista di record giornalieri con tutte le feature.
    Ogni record è pronto per query/filtering esterno.
    """
    records = []
    for dt, row in df_fwd.iterrows():
        rec = {
            "date":             dt.strftime("%Y-%m-%d"),
            "close":            _safe_float(row.get("price")),
            "log_ret":          _safe_float(row.get("log_ret")),
            "rolling_mean":     _safe_float(row.get("rolling_mean")),
            "expanding_mean":   _safe_float(row.get("expanding_mean")),
            "diff":             _safe_float(row.get("diff")),
            "abs_diff":         _safe_float(abs(row.get("diff", np.nan))),
            "threshold":        round(threshold, 8),
            "is_non_ergodic":   bool(row.get("is_non_ergodic", False)),
            "regime":           "non_ergodico" if row.get("is_non_ergodic") else "ergodico",
        }
        # Forward returns (NaN → None per JSON pulito)
        for label in _FWD_WINDOWS:
            rec[f"fwd_{label}"] = _safe_float(row.get(f"fwd_{label}"))
        records.append(rec)
    return records


# ================================================================
# SEZIONE 2 — NON-ERGODIC RUNS (cluster continui)
# ================================================================

def _build_non_ergodic_runs(df_fwd: pd.DataFrame) -> list[dict]:
    """
    Identifica i periodi continui di non-ergodicità e calcola per ciascuno:
    durata, diff massima, rendimento ex-post 1M/3M/6M/12M dalla fine del run.
    """
    runs = []
    in_run = False
    run_start = None
    run_rows = []

    for dt, row in df_fwd.iterrows():
        is_ne = bool(row.get("is_non_ergodic", False))
        if is_ne and not in_run:
            in_run = True
            run_start = dt
            run_rows = [row]
        elif is_ne and in_run:
            run_rows.append(row)
        elif not is_ne and in_run:
            # Fine del run
            run_end = dt
            run_df = pd.DataFrame(run_rows)
            fwd_at_exit = {
                f"fwd_{lbl}": _safe_float(row.get(f"fwd_{lbl}"))
                for lbl in _FWD_WINDOWS
            }
            runs.append({
                "start":          run_start.strftime("%Y-%m-%d"),
                "end":            run_end.strftime("%Y-%m-%d"),
                "duration_days":  len(run_rows),
                "max_abs_diff":   _safe_float(run_df["diff"].abs().max()),
                "mean_abs_diff":  _safe_float(run_df["diff"].abs().mean()),
                "direction":      "positive" if float(run_df["diff"].mean()) > 0 else "negative",
                **fwd_at_exit,   # rendimento misurato dalla fine del run
            })
            in_run = False
            run_rows = []

    return runs


# ================================================================
# SEZIONE 3 — REGIME ALPHA (statistiche forward-return per regime)
# ================================================================

def _build_regime_alpha(df_fwd: pd.DataFrame) -> dict:
    """
    Per ciascun regime (ergodico / non_ergodico) e per ciascun orizzonte
    calcola: n, mean, std, hit_rate, sharpe.
    """
    result = {}
    for regime_val, regime_label in [(False, "ergodico"), (True, "non_ergodico")]:
        sub = df_fwd[df_fwd["is_non_ergodic"] == regime_val]
        regime_data: dict[str, dict] = {}
        for label in _FWD_WINDOWS:
            col = f"fwd_{label}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            regime_data[label] = {
                "n":        int(len(s)),
                "mean_pct": _safe_float(s.mean()),
                "std_pct":  _safe_float(s.std()),
                "hit_rate": _hit_rate(s),
                "sharpe":   _sharpe(s),
                "p10":      _safe_float(s.quantile(0.10)),
                "median":   _safe_float(s.median()),
                "p90":      _safe_float(s.quantile(0.90)),
            }
        result[regime_label] = regime_data
    return result


# ================================================================
# SEZIONE 4 — ALPHA SIGNALS (transizioni di regime)
# ================================================================

def _build_alpha_signals(df_fwd: pd.DataFrame, threshold: float) -> list[dict]:
    """
    Pre-filtra i giorni di transizione di regime come segnali operativi:
      - NON_ERGODIC_ENTRY : primo giorno non ergodico dopo un periodo ergodico
      - NON_ERGODIC_EXIT  : primo giorno ergodico dopo un periodo non ergodico
      - EXTREME_DIFF_HIGH : |diff| > 2 × threshold (stress estremo)
      - EXTREME_DIFF_LOW  : |diff| < 0.1 × threshold (compressione)
    """
    signals = []
    prev_ne = None

    for dt, row in df_fwd.iterrows():
        is_ne = bool(row.get("is_non_ergodic", False))
        abs_diff = abs(float(row.get("diff", 0)))

        signal_type = None
        if prev_ne is not None:
            if is_ne and not prev_ne:
                signal_type = "NON_ERGODIC_ENTRY"
            elif not is_ne and prev_ne:
                signal_type = "NON_ERGODIC_EXIT"

        if abs_diff > 2 * threshold:
            signal_type = "EXTREME_DIFF_HIGH"
        elif abs_diff < 0.1 * threshold:
            signal_type = "EXTREME_DIFF_LOW"

        if signal_type:
            rec = {
                "date":        dt.strftime("%Y-%m-%d"),
                "signal":      signal_type,
                "diff":        _safe_float(row.get("diff")),
                "abs_diff":    _safe_float(abs_diff),
                "threshold":   round(threshold, 8),
                "stress_ratio": _safe_float(abs_diff / threshold) if threshold else None,
            }
            for label in _FWD_WINDOWS:
                rec[f"fwd_{label}"] = _safe_float(row.get(f"fwd_{label}"))
            signals.append(rec)

        prev_ne = is_ne

    return signals


# ================================================================
# SEZIONE 5 — DECADE STATS
# ================================================================

def _build_decade_stats(decade_df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in decade_df.iterrows():
        records.append({
            "decade":             str(row.get("decade", "")),
            "total_days":         int(row.get("giorni_totali", 0)),
            "non_ergodic_days":   int(row.get("giorni_non_ergodici", 0)),
            "pct_non_ergodic":    _safe_float(row.get("pct_non_ergodici")),
            "mean_diff":          _safe_float(row.get("diff_medio")),
            "std_diff":           _safe_float(row.get("diff_std")),
        })
    return records


# ================================================================
# SEZIONE 6 — STATISTICAL SUMMARY
# ================================================================

def _build_statistical_summary(df_fwd: pd.DataFrame, diff_stats: dict) -> dict:
    diff = df_fwd["diff"].dropna()
    autocorr = {
        f"lag_{lag}": _safe_float(diff.autocorr(lag=lag))
        for lag in [1, 5, 21, 63]
    }
    return {
        "diff_distribution": {
            "mean":        _safe_float(diff_stats.get("media")),
            "std":         _safe_float(diff_stats.get("std")),
            "skewness":    _safe_float(diff_stats.get("skewness")),
            "kurtosis_excess": _safe_float(diff_stats.get("kurtosis_excess")),
            "p5":          _safe_float(diff_stats.get("percentile_5")),
            "p95":         _safe_float(diff_stats.get("percentile_95")),
            "min":         _safe_float(diff_stats.get("minimo")),
            "max":         _safe_float(diff_stats.get("massimo")),
        },
        "autocorrelation_diff": autocorr,
        "pct_positive_diff": _safe_float((diff > 0).mean() * 100),
    }


# ================================================================
# ENTRY POINT
# ================================================================

def build_ergodicity_export(
    result: ErgodicityResult,
    decade_stats: pd.DataFrame,
    diff_stats: dict,
    ticker: str,
    ticker_label: str,
) -> dict:
    """
    Costruisce il dizionario completo da serializzare come JSON.

    Parameters
    ----------
    result       : ErgodicityResult dall'analisi principale
    decade_stats : DataFrame da compute_decade_stats()
    diff_stats   : Dict da compute_diff_statistics()
    ticker       : Ticker EODHD (es. 'GSPC.INDX')
    ticker_label : Nome leggibile (es. 'S&P 500')

    Returns
    -------
    dict serializzabile con json.dumps()
    """
    df_fwd = _add_forward_returns(result.df)

    return {
        "metadata": {
            "export_date":       date.today().isoformat(),
            "study":             "ergodicity_market",
            "ticker":            ticker,
            "ticker_label":      ticker_label,
            "start_date":        result.df.index[0].strftime("%Y-%m-%d"),
            "end_date":          result.df.index[-1].strftime("%Y-%m-%d"),
            "n_observations":    result.n_total,
            "parameters": {
                "rolling_window":    int(result.df["rolling_mean"].rolling(2).count().max()),
                "threshold_mode":    "sem",
                "k_multiplier":      _safe_float(result.k_mult),
                "threshold":         _safe_float(result.threshold),
                "sigma_global":      _safe_float(result.sigma_global),
                "sem":               _safe_float(result.sem),
            },
            "summary": {
                "n_non_ergodic":     result.n_non_ergodic,
                "pct_non_ergodic":   _safe_float(result.pct_non_ergodic),
                "current_diff":      _safe_float(result.current_diff),
                "is_ergodic_now":    result.is_ergodic_now,
                "status_label":      result.status_label,
            },
        },
        "time_series":          _build_time_series(df_fwd, result.threshold),
        "non_ergodic_runs":     _build_non_ergodic_runs(df_fwd),
        "regime_alpha":         _build_regime_alpha(df_fwd),
        "alpha_signals":        _build_alpha_signals(df_fwd, result.threshold),
        "decade_stats":         _build_decade_stats(decade_stats),
        "statistical_summary":  _build_statistical_summary(df_fwd, diff_stats),
    }
