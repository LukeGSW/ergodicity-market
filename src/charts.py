"""
charts.py — Funzioni per la creazione dei grafici Plotly dell'analisi di ergodicità.

Ogni funzione restituisce un go.Figure pronto per st.plotly_chart().
La palette colori e il layout base sono fedeli all'app originale (dark theme).

Grafici implementati:
  1. build_price_chart()         → Prezzo (scala log) con eventi non ergodici
  2. build_means_chart()         → Media temporale vs spaziale + banda ergodicità
  3. build_diff_histogram()      → Istogramma differenza (rolling − expanding) + soglie
  4. build_rolling_pct_chart()   → % giorni non ergodici nel tempo (rolling 252g)
  5. build_decade_bar_chart()    → Barplot % non ergodici per decennio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.calculations import ErgodicityResult, TRADING_DAYS_YEAR


# ============================================================
# PALETTE COLORI (fedele all'app originale)
# ============================================================

COLORS = {
    "price_line":   "#3F51B5",   # indaco — linea prezzo graf.1
    "non_ergodic":  "#F44336",   # rosso — punti non ergodici
    "rolling":      "#00BCD4",   # ciano — media temporale (rolling) graf.2
    "expanding":    "#FF9800",   # arancio — media spaziale (expanding) graf.2
    "threshold_g2": "#4CAF50",   # verde — soglie banda ergodicità graf.2
    "threshold_g3": "#F44336",   # rosso — soglie istogramma graf.3
    "histogram":    "#2196F3",   # blu — barre istogramma
    "accent":       "#AB47BC",   # viola — grafico % non ergodici
    "neutral":      "#9E9E9E",   # grigio — linee di riferimento
    "background":   "#0D0D0D",   # sfondo
    "surface":      "#161616",   # pannello grafico
    "text":         "#E0E0E0",   # testo
    "grid":         "#222233",   # griglia
}


# ============================================================
# LAYOUT BASE CONDIVISO
# ============================================================

def _base_layout(
    title: str,
    x_title: str = "Data",
    y_title: str = "",
    height: int = 450,
) -> dict:
    """
    Restituisce il dizionario di layout Plotly standard condiviso da tutti i grafici.

    Args:
        title:   Titolo del grafico
        x_title: Label asse X
        y_title: Label asse Y
        height:  Altezza in pixel

    Returns:
        dict pronto per fig.update_layout(**layout)
    """
    return dict(
        title=dict(text=title, font=dict(size=15, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            color=COLORS["text"],
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            color=COLORS["text"],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333355",
            font=dict(color=COLORS["text"]),
        ),
        hovermode="x unified",
        margin=dict(l=70, r=30, t=65, b=55),
        height=height,
    )


# ============================================================
# GRAFICO 1 — PREZZO (scala log) + eventi non ergodici
# ============================================================

def build_price_chart(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    """
    Grafico 1: prezzo reale in scala logaritmica con punti rossi
    sovrapposti nei giorni non ergodici.

    L'asse Y usa scala log (non log del prezzo) per rendere comparabili
    le variazioni percentuali su orizzonti storici molto lunghi (es. 1950→oggi).

    Args:
        result:       ErgodicityResult da calculations.py
        ticker_label: Etichetta breve (es. 'SPX')

    Returns:
        go.Figure
    """
    df = result.df
    df_ne = df[df["is_non_ergodic"]]

    fig = go.Figure()

    # Linea prezzo reale
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["price"],
        name="Prezzo",
        line=dict(color=COLORS["price_line"], width=1.0),
        hovertemplate="%{x|%Y-%m-%d}<br>Prezzo: %{y:,.2f}<extra></extra>",
    ))

    # Scatter eventi non ergodici sovrapposti
    fig.add_trace(go.Scatter(
        x=df_ne.index,
        y=df_ne["price"],
        name="Eventi non ergodici",
        mode="markers",
        marker=dict(
            color=COLORS["non_ergodic"],
            size=5,
            opacity=0.85,
            symbol="circle",
        ),
        hovertemplate="%{x|%Y-%m-%d}<br><b>Non ergodico</b><extra></extra>",
    ))

    layout = _base_layout(
        title=f"{ticker_label} – Prezzo (log) con evidenza eventi non ergodici",
        x_title="Data",
        y_title="Prezzo",
        height=460,
    )
    # Sovrascrive yaxis per scala logaritmica
    layout["yaxis"]["type"] = "log"
    layout["yaxis"]["dtick"] = 1   # tick ogni ordine di grandezza
    fig.update_layout(**layout)

    return fig


# ============================================================
# GRAFICO 2 — MEDIA TEMPORALE vs SPAZIALE + BANDA ERGODICITÀ
# ============================================================

def build_means_chart(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    """
    Grafico 2 — Media temporale vs media d'insieme.

    In modalità 'vol_drag' mostra la media d'insieme adattiva (aritmetica) e la
    media temporale (geometrica): l'area tra le due È il volatility drag
    (non-ergodicità). I giorni di eccesso (z > k) sono evidenziati in rosso.

    In modalità 'drift_divergence' (legacy) mostra rolling vs expanding mean dei
    log-return con banda di ergodicità fissa a ±threshold.

    Args:
        result:       ErgodicityResult da calculations.py
        ticker_label: Etichetta breve (es. 'SPX')

    Returns:
        go.Figure
    """
    if result.method == "vol_drag":
        return _build_means_chart_vol_drag(result, ticker_label)
    return _build_means_chart_legacy(result, ticker_label)


def _build_means_chart_vol_drag(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    df = result.df
    df_ne = df[df["is_non_ergodic"]]

    fig = go.Figure()

    # Media temporale (geometrica) — ciano: la base su cui si appoggia il drag
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["rolling_mean"],
        name="Media temporale (geometrica)",
        line=dict(color=COLORS["rolling"], width=1.2),
        hovertemplate="g_tempo: %{y:.6f}<extra></extra>",
    ))

    # Media d'insieme (aritmetica, adattiva) — arancio; fill = volatility drag
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["ensemble_mean"],
        name="Media d'insieme (aritmetica)",
        line=dict(color=COLORS["expanding"], width=1.4),
        fill="tonexty",
        fillcolor="rgba(255,152,0,0.14)",
        hovertemplate="g_insieme: %{y:.6f}<extra></extra>",
    ))

    # Eccessi non ergodici (z > k) — punti rossi sulla media d'insieme
    fig.add_trace(go.Scatter(
        x=df_ne.index,
        y=df_ne["ensemble_mean"],
        name=f"Eccesso non ergodico (z>{result.threshold:.2f})",
        mode="markers",
        marker=dict(color=COLORS["non_ergodic"], size=4, opacity=0.8),
        hovertemplate="%{x|%Y-%m-%d}<br><b>Eccesso non ergodico</b><extra></extra>",
    ))

    layout = _base_layout(
        title=f"{ticker_label} – Media temporale vs d'insieme — l'area è il volatility drag",
        x_title="Data",
        y_title="Crescita per-periodo (rendimento semplice)",
        height=460,
    )
    fig.update_layout(**layout)
    return fig


def _build_means_chart_legacy(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    df = result.df
    thr = result.threshold

    fig = go.Figure()

    # Banda ergodicità: fill tra soglia- e soglia+ (verde trasparente)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[-thr] * len(df),
        name=f"Soglia –",
        line=dict(color=COLORS["threshold_g2"], width=1.2, dash="dash"),
        mode="lines",
        hovertemplate=f"Soglia –: {-thr:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[thr] * len(df),
        name=f"Soglia +",
        line=dict(color=COLORS["threshold_g2"], width=1.2, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(76,175,80,0.10)",
        mode="lines",
        hovertemplate=f"Soglia +: {thr:.6f}<extra></extra>",
    ))

    # Media spaziale (expanding) — arancio punteggiato
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["expanding_mean"],
        name="Media spaziale (expanding)",
        line=dict(color=COLORS["expanding"], width=1.6, dash="dot"),
        hovertemplate="Media spaziale: %{y:.6f}<extra></extra>",
    ))

    # Media temporale (rolling) — ciano
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["rolling_mean"],
        name="Media temporale (rolling)",
        line=dict(color=COLORS["rolling"], width=1.0),
        hovertemplate="Media rolling: %{y:.6f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=f"{ticker_label} – Medie dei rendimenti logaritmici e banda di ergodicità",
        x_title="Data",
        y_title="Valore (log returns — media)",
        height=460,
    ))
    return fig


# ============================================================
# GRAFICO 3 — DISTRIBUZIONE DIFFERENZA (rolling − expanding)
# ============================================================

def build_diff_histogram(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    """
    Grafico 3: istogramma della differenza (rolling_mean − expanding_mean)
    con linee verticali rosse tratteggiate alle soglie di ergodicità.

    La distribuzione rivela:
    - Centratura intorno a 0 (mercato mediamente ergodico)
    - Asimmetria (skewness): bias direzionale nella deviazione
    - Pesantezza delle code (kurtosis): frequenza dei regimi estremi

    Args:
        result:       ErgodicityResult da calculations.py
        ticker_label: Etichetta breve (es. 'SPX')

    Returns:
        go.Figure
    """
    thr = result.threshold
    is_vd = result.method == "vol_drag"

    # In vol_drag l'istogramma è sullo z-score (segnale di rilevazione), con
    # un'unica soglia positiva (one-sided). In legacy è sulla differenza Δ con
    # soglie simmetriche ±threshold.
    signal = (result.df["z_score"] if is_vd else result.df["diff"]).dropna()
    x_title = "z-score (eccesso di drag)" if is_vd else "Differenza"
    title = (
        "Distribuzione dello z-score del volatility drag con soglia"
        if is_vd else
        "Distribuzione della differenza (rolling − expanding) con soglie"
    )
    fmt = ":.3f" if is_vd else ":.6f"

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=signal,
        name="Segnale",
        nbinsx=100,
        marker_color=COLORS["histogram"],
        opacity=0.88,
        hovertemplate=f"Valore: %{{x{fmt}}}<br>Frequenza: %{{y}}<extra></extra>",
    ))

    if is_vd:
        # Soglia singola: eccesso ⟺ z > k
        fig.add_vline(
            x=thr,
            line_color=COLORS["threshold_g3"],
            line_dash="dash",
            line_width=2,
            annotation_text=f"soglia z={thr:.2f}",
            annotation_position="top right",
            annotation_font_color=COLORS["threshold_g3"],
            annotation_font_size=11,
        )
    else:
        # Soglie simmetriche ±threshold
        fig.add_vline(
            x=-thr,
            line_color=COLORS["threshold_g3"],
            line_dash="dash",
            line_width=2,
            annotation_text="–soglia",
            annotation_position="top left",
            annotation_font_color=COLORS["threshold_g3"],
            annotation_font_size=11,
        )
        fig.add_vline(
            x=thr,
            line_color=COLORS["threshold_g3"],
            line_dash="dash",
            line_width=2,
            annotation_text="+soglia",
            annotation_position="top right",
            annotation_font_color=COLORS["threshold_g3"],
            annotation_font_size=11,
        )

    layout = _base_layout(
        title=title,
        x_title=x_title,
        y_title="Frequenza",
        height=400,
    )
    layout["showlegend"] = False
    layout["bargap"] = 0.02
    fig.update_layout(**layout)

    return fig


# ============================================================
# GRAFICO 4 — % GIORNI NON ERGODICI NEL TEMPO (rolling 252g)
# ============================================================

def build_rolling_pct_chart(result: ErgodicityResult, ticker_label: str) -> go.Figure:
    """
    Grafico 4 (bonus): percentuale rolling di giorni non ergodici
    calcolata su finestra di 252 giorni (1 anno trading).

    Utile per identificare i periodi storici di maggiore instabilità
    ergodica (crisi finanziarie, cambi di regime, shock esogeni).

    Args:
        result:       ErgodicityResult da calculations.py
        ticker_label: Etichetta breve (es. 'SPX')

    Returns:
        go.Figure
    """
    df = result.df
    win = result.rolling_window
    rolling_pct = df["is_non_ergodic"].rolling(win).mean() * 100
    avg_pct = result.pct_non_ergodic

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_pct.index,
        y=rolling_pct,
        name=f"% non ergodici (rolling {win}g)",
        fill="tozeroy",
        line=dict(color=COLORS["accent"], width=1.4),
        fillcolor="rgba(171,71,188,0.18)",
        hovertemplate="%{x|%Y-%m-%d}<br>% non ergodici: %{y:.1f}%<extra></extra>",
    ))

    # Linea media storica
    fig.add_hline(
        y=avg_pct,
        line_dash="dash",
        line_color=COLORS["neutral"],
        line_width=1.5,
        annotation_text=f"Media storica {avg_pct:.1f}%",
        annotation_position="bottom right",
        annotation_font_color=COLORS["neutral"],
        annotation_font_size=11,
    )

    layout = _base_layout(
        title=f"{ticker_label} – % Giorni Non Ergodici (rolling {win}g)",
        x_title="Data",
        y_title="% Non Ergodici",
        height=380,
    )
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig


# ============================================================
# GRAFICO 5 — BARPLOT % NON ERGODICI PER DECENNIO
# ============================================================

def build_decade_bar_chart(decade_stats: pd.DataFrame) -> go.Figure:
    """
    Grafico 5: barplot della percentuale di giorni non ergodici per decennio.

    Permette di visualizzare a colpo d'occhio i decenni storicamente
    più instabili dal punto di vista dell'ergodicità.

    Args:
        decade_stats: DataFrame prodotto da compute_decade_stats()

    Returns:
        go.Figure
    """
    pct = decade_stats["pct_non_ergodici"]
    avg = pct.mean()

    # Colore barre: rosso se sopra media, verde se sotto
    bar_colors = [
        COLORS["non_ergodic"] if v > avg else COLORS["threshold_g2"]
        for v in pct
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decade_stats["decade"],
        y=pct,
        marker_color=bar_colors,
        opacity=0.85,
        text=[f"{v:.1f}%" for v in pct],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="%{x}<br>% Non ergodici: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=avg,
        line_dash="dash",
        line_color=COLORS["neutral"],
        line_width=1.5,
        annotation_text=f"Media {avg:.1f}%",
        annotation_font_color=COLORS["neutral"],
    )

    layout = _base_layout(
        title="% Giorni Non Ergodici per Decennio",
        x_title="Decennio",
        y_title="% Giorni Non Ergodici",
        height=380,
    )
    layout["yaxis"]["ticksuffix"] = "%"
    layout["showlegend"] = False
    fig.update_layout(**layout)

    return fig
