"""
app.py — Entry point dell'app Streamlit: Ergodicità del Mercato.

Struttura dell'app:
  - Sidebar: selezione asset, data inizio, finestra rolling, soglia
  - Main:
      [0] Header educativo + concetto di ergodicità
      [1] KPI: soglia, % non ergodici, stato attuale, diff attuale
      [2] Grafico 1 — Prezzo (log) con eventi non ergodici
      [3] Grafico 2 — Media temporale vs spaziale + banda ergodicità
      [4] Grafico 3 — Distribuzione differenza (rolling − expanding)
      [5] Grafico 4 — % giorni non ergodici nel tempo
      [6] Grafico 5 — Barplot % per decennio
      [7] Tabella statistiche per decennio
      [8] Expander metodologia e riferimenti
"""

import streamlit as st
import pandas as pd
from datetime import date

from src.data_fetcher import fetch_ohlcv, resolve_ticker, TICKER_MAP
from src.calculations import (
    compute_ergodicity_metrics,
    compute_decade_stats,
    compute_diff_statistics,
)
from src.charts import (
    build_price_chart,
    build_means_chart,
    build_diff_histogram,
    build_rolling_pct_chart,
    build_decade_bar_chart,
)


# ============================================================
# CONFIGURAZIONE PAGINA
# ============================================================
st.set_page_config(
    page_title="Ergodicità del Mercato | Kriterion Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# API KEY (da Streamlit Secrets)
# ============================================================
try:
    EODHD_API_KEY = st.secrets["EODHD_API_KEY"]
except KeyError:
    st.error(
        "❌ **API Key mancante.** Aggiungi `EODHD_API_KEY` nei Secrets dell'app "
        "(Settings → Secrets su Streamlit Cloud, oppure `.streamlit/secrets.toml` in locale)."
    )
    st.stop()


# ============================================================
# SIDEBAR — PARAMETRI UTENTE
# ============================================================
with st.sidebar:
    st.title("⚙️ Parametri")
    st.divider()

    # --- Scorciatoie ticker ---
    st.markdown("**Scorciatoie:**")
    shortcuts = list(TICKER_MAP.keys())
    cols = st.columns(4)
    selected_shortcut = None
    for i, s in enumerate(shortcuts):
        if cols[i % 4].button(s, key=f"btn_{s}", width="stretch"):
            selected_shortcut = s

    # --- Input ticker manuale ---
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = "GSPC.INDX"
    if selected_shortcut:
        eodhd_ticker, ticker_label = resolve_ticker(selected_shortcut)
        st.session_state["ticker_input"] = eodhd_ticker
        st.session_state["ticker_label"] = ticker_label
    else:
        if "ticker_label" not in st.session_state:
            st.session_state["ticker_label"] = "SPX"

    ticker_raw = st.text_input(
        "oppure inserisci il ticker EODHD:",
        value=st.session_state["ticker_input"],
        help="Formato: GSPC.INDX, SPY.US, BTC-USD.CC, GC.COMM …",
        key="ticker_text_input",
    )
    eodhd_ticker, ticker_label = resolve_ticker(ticker_raw)

    st.divider()

    # --- Data di partenza ---
    start_date = st.date_input(
        "Data di partenza",
        value=date(1950, 1, 1),
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        help="I dati disponibili variano per asset. Per indici US sono disponibili dal 1950.",
    )

    st.divider()

    # --- Parametri ergodicità ---
    st.markdown("**Parametri ergodicità:**")

    rolling_window = st.slider(
        "Finestra rolling (giorni)",
        min_value=21,
        max_value=756,
        value=252,
        step=21,
        help="252 = 1 anno di trading. Finestre più corte sono più reattive ma meno stabili.",
    )

    threshold_mode = st.radio(
        "Calcolo soglia",
        options=["sem", "manual"],
        index=0,
        format_func=lambda x: {
            "sem":    "SEM — k × σ / √N  (raccomandato)",
            "manual": "Manuale — valore fisso",
        }[x],
        help=(
            "**SEM**: Standard Error of the Mean = k × σ_globale / √N. "
            "Fondamento statistico rigoroso: la rolling mean ha incertezza σ/√N. "
            "Deviazioni oltre questa soglia sono statisticamente significative.\n\n"
            "**Manuale**: inserisci un valore fisso (utile per confronto tra asset)."
        ),
    )

    threshold_mult = 1.75
    manual_threshold = 0.0011

    if threshold_mode == "sem":
        threshold_mult = st.select_slider(
            "Moltiplicatore k",
            options=[1.00, 1.28, 1.50, 1.65, 1.75, 1.96, 2.00, 2.33, 2.58],
            value=1.75,
            format_func=lambda x: {
                1.00: "1.00  (68% CI — 1σ)",
                1.28: "1.28  (80% CI)",
                1.50: "1.50  (87% CI)",
                1.65: "1.65  (90% CI)",
                1.75: "1.75  (92% CI — default)",
                1.96: "1.96  (95% CI — standard scientifico)",
                2.00: "2.00  (95.4% CI)",
                2.33: "2.33  (98% CI)",
                2.58: "2.58  (99% CI — molto conservativo)",
            }.get(x, str(x)),
            help=(
                "k determina il livello di confidenza statistica della classificazione. "
                "k=1.75 replica il valore ±0.0011 osservato per SPX/252g. "
                "k=1.96 è il valore standard per test a 95% di confidenza."
            ),
        )
    else:
        manual_threshold = st.number_input(
            "Soglia manuale",
            min_value=0.00001,
            max_value=0.1,
            value=0.0011,
            step=0.0001,
            format="%.6f",
            help="Valore fisso per |rolling_mean − expanding_mean|. Es: 0.0011 per SPX/252g.",
        )

    st.divider()
    st.caption("📡 Dati: EODHD Historical Data")
    st.caption("🛠️ Kriterion Quant — Finanza Quantitativa")


# ============================================================
# SEZIONE 0 — HEADER E INTRO EDUCATIVA
# ============================================================
st.title("📊 Ergodicità del Mercato")
st.markdown("### Rolling Mean vs Global Mean — Studio Statistico")

with st.expander("📖 Cos'è l'Ergodicità in Finanza? (leggi prima di procedere)", expanded=False):
    st.markdown("""
    ## Il Problema dell'Ergodicità in Economia

    Il concetto di **ergodicità** origina dalla fisica statistica e termodinamica: un sistema
    fisico è ergodico quando la *media temporale* (calcolata su un singolo percorso nel tempo)
    converge alla *media d'insieme* (calcolata su tutti i possibili stati del sistema nello stesso
    istante).

    ### Perché è importante in finanza?

    In finanza, la distinzione è cruciale:
    - **Media temporale**: "quanto ha reso *questo* asset negli ultimi N anni?"
    - **Media d'insieme** (ensemble): "quanto renderebbe *in media* questo tipo di asset?"

    Se queste due medie divergono sistematicamente, il mercato è **non ergodico** in quel
    periodo: le aspettative ricavate dalla storia recente non riflettono più il comportamento
    di lungo periodo dell'asset.

    ### Riferimenti teorici

    - **Ole Peters (2019)** — *"The ergodicity problem in economics"*, Nature Physics 15.
      Peters dimostra che molti modelli economici standard assumono erroneamente l'ergodicità,
      portando a conclusioni sistematicamente errate sulla gestione del rischio.
    - **Nassim N. Taleb** — *Incerto* series. Taleb identifica la non-ergodicità come uno dei
      meccanismi fondamentali della fragilità finanziaria.

    ### Come funziona questa analisi

    1. **Log-return** → `r_t = ln(P_t / P_{t-1})`: metrica additiva e simmetrica
    2. **Media temporale (rolling)** → `μ_T(t) = mean(r_{t-N}, ..., r_t)`: media locale degli ultimi N giorni
    3. **Media spaziale (expanding)** → `μ_E(t) = mean(r_1, ..., r_t)`: media globale dall'origine
    4. **Differenza** → `Δ(t) = μ_T(t) − μ_E(t)`: scostamento locale dalla media storica
    5. **Soglia SEM** → `threshold = k × σ / √N`: se `|Δ(t)| > threshold`, il giorno è **non ergodico**

    ### Fondamento Statistico della Soglia (Standard Error of the Mean)

    La rolling mean calcolata su N osservazioni ha un'incertezza statistica di **σ/√N**
    (Standard Error of the Mean). Questo significa che, anche se il processo fosse
    perfettamente ergodico, la rolling mean oscillerebbe attorno alla vera media con
    questa ampiezza per puro effetto campionario.

    Classificare come "non ergodico" una deviazione superiore a **k × σ/√N** equivale
    a un test statistico sulla media con livello di confidenza associato a k:

    | k | Confidenza | Interpretazione |
    |---|------------|-----------------|
    | 1.00 | 68.3% | Soglia molto sensibile (molti falsi positivi) |
    | 1.75 | ~92% | Default — replica ±0.0011 per SPX/252g |
    | 1.96 | 95.0% | Standard scientifico per test di ipotesi |
    | 2.58 | 99.0% | Soglia conservativa (solo deviazioni estreme) |

    > **Proprietà chiave:** la soglia SEM si restringe automaticamente aumentando N
    > (finestre più lunghe → stime più precise → banda più stretta) e si allarga per
    > asset più volatili (σ maggiore). Questo comportamento è fisicamente corretto
    > e non si ottiene con approcci basati su std(diff).
    """)

st.divider()


# ============================================================
# FETCH DATI
# ============================================================
end_date_str = date.today().strftime("%Y-%m-%d")
start_date_str = start_date.strftime("%Y-%m-%d")

with st.spinner(f"⏳ Caricamento dati per **{ticker_label}** ({eodhd_ticker})..."):
    try:
        df_raw = fetch_ohlcv(eodhd_ticker, start_date_str, end_date_str, EODHD_API_KEY)
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dati: {e}")
        st.info(
            "💡 Suggerimento: verifica che il ticker sia nel formato EODHD corretto "
            "(es. `GSPC.INDX` per S&P 500, `SPY.US` per ETF, `BTC-USD.CC` per Bitcoin)."
        )
        st.stop()

if df_raw.empty:
    st.warning(f"⚠️ Nessun dato trovato per **{eodhd_ticker}**. Verifica ticker e date.")
    st.stop()


# ============================================================
# CALCOLO ERGODICITÀ
# ============================================================
with st.spinner("⚙️ Calcolo metriche di ergodicità..."):
    result = compute_ergodicity_metrics(
        df=df_raw,
        rolling_window=rolling_window,
        threshold_mode=threshold_mode,
        threshold_mult=threshold_mult,
        manual_threshold=manual_threshold,
    )
    decade_stats = compute_decade_stats(result)
    diff_stats = compute_diff_statistics(result)


# ============================================================
# SEZIONE 1 — KPI METRICHE
# ============================================================
st.subheader(f"📋 Riepilogo — {ticker_label}")
st.markdown(
    f"**{ticker_label}** &nbsp;•&nbsp; Finestra: {rolling_window}g &nbsp;•&nbsp; "
    f"Soglia: ±{result.threshold:.4f} &nbsp;•&nbsp; "
    f"Giorni non ergodici: {result.pct_non_ergodic:.1f}% &nbsp;•&nbsp; "
    f"Stato attuale: **{result.status_label}**"
)

# Riga 1: parametri della soglia SEM
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Finestra rolling N",
    f"{rolling_window}g",
    help="Numero di giorni per la media temporale (rolling mean)",
)
c2.metric(
    "σ globale (log-return)",
    f"{result.sigma_global:.5f}",
    help="Deviazione standard dei log-return sull'intera storia disponibile",
)
c3.metric(
    "SEM = σ / √N",
    f"{result.sem:.5f}",
    help="Standard Error of the Mean: incertezza statistica della rolling mean su N osservazioni",
)
c4.metric(
    "Soglia = k × SEM",
    f"±{result.threshold:.5f}",
    help=f"k={result.k_mult:.2f} → confidenza ~{min(99.9, 100*(1 - 2*(1-0.5*(1+__import__('math').erf(result.k_mult/2**0.5))))):.1f}%",
)

st.markdown("")  # piccolo spazio

# Riga 2: risultati
r1, r2, r3, r4 = st.columns(4)
r1.metric(
    "Giorni analizzati",
    f"{result.n_total:,}",
    help="Giorni totali con dati sufficienti per calcolare rolling e expanding mean",
)
r2.metric(
    "Giorni non ergodici",
    f"{result.n_non_ergodic:,}",
    delta=f"{result.pct_non_ergodic:.1f}% del totale",
    delta_color="inverse" if result.pct_non_ergodic > 15 else "normal",
)
r3.metric(
    "Differenza attuale",
    f"{result.current_diff:.5f}",
    delta=f"{'DENTRO' if result.is_ergodic_now else 'FUORI'} banda",
    delta_color="normal" if result.is_ergodic_now else "inverse",
)
r4.metric(
    "Stato attuale",
    "ERGODICO ✅" if result.is_ergodic_now else "NON ERGODICO ⚠️",
    help="Basato sull'ultimo valore disponibile della differenza",
)

st.divider()


# ============================================================
# SEZIONE 2 — GRAFICO 1: PREZZO (LOG SCALE)
# ============================================================
st.subheader("📈 Grafico 1 — Prezzo con Evidenza degli Eventi Non Ergodici")
st.markdown(f"""
Il grafico mostra il prezzo di **{ticker_label}** in **scala logaritmica** dal
`{result.df.index[0].date()}` ad oggi. I **punti rossi** indicano i giorni in cui la
differenza tra media rolling e media expanding ha superato la soglia di ergodicità.

> 📌 **Perché la scala logaritmica?** Su orizzonti storici lunghi, i prezzi crescono
> esponenzialmente. La scala log trasforma la crescita esponenziale in una linea quasi retta,
> rendendo visivamente comparabili variazioni percentuali di decenni diversi. Una variazione
> di +10% nel 1960 è visivamente identica a una di +10% nel 2020.
""")

fig1 = build_price_chart(result, ticker_label)
st.plotly_chart(fig1, width="stretch")

st.divider()


# ============================================================
# SEZIONE 3 — GRAFICO 2: MEDIE + BANDA ERGODICITÀ
# ============================================================
st.subheader("📊 Grafico 2 — Media Temporale vs Media Spaziale")
st.markdown(f"""
Questo è il **grafico centrale** dell'analisi di ergodicità.

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 🔵 Ciano | Media temporale (rolling) | Media dei log-return degli ultimi {rolling_window} giorni |
| 🟠 Arancio punteggiato | Media spaziale (expanding) | Media cumulativa dall'origine (stima del "vero" rendimento atteso) |
| 🟢 Verde tratteggiato | Soglie ±{result.threshold:.5f} | Banda di ergodicità: dentro = ergodico, fuori = non ergodico |

**Interpretazione:**
- Quando la linea **ciano** rimane all'interno della banda verde, il mercato è ergodico:
  le aspettative di breve e lungo periodo sono allineate.
- Quando la ciano **esce dalla banda**, il mercato è in un regime transitorio: la storia
  recente diverge dalla distribuzione di lungo periodo (possibile cambio di regime,
  shock macro, crisi).

> 📌 **Nota statistica:** la media spaziale (expanding) scende lentamente verso zero
> nel tempo perché accumula decenni di dati. I mercati azionari mostrano rendimenti
> medi giornalieri nell'ordine di **0.03% — 0.05%** su orizzonti molto lunghi.
""")

fig2 = build_means_chart(result, ticker_label)
st.plotly_chart(fig2, width="stretch")

st.divider()


# ============================================================
# SEZIONE 4 — GRAFICO 3: DISTRIBUZIONE DIFFERENZE
# ============================================================
st.subheader("📉 Grafico 3 — Distribuzione della Differenza (Rolling − Expanding)")
st.markdown(f"""
L'istogramma mostra la **distribuzione statistica** di tutte le differenze
`Δ(t) = rolling_mean(t) − expanding_mean(t)` calcolate sull'intera serie storica.

- Le **linee rosse tratteggiate** segnano la soglia ±{result.threshold:.5f}
- Le barre **fuori dalla banda rossa** corrispondono ai giorni non ergodici
- La forma della distribuzione fornisce informazioni diagnostiche importanti:

| Metrica | Valore | Interpretazione |
|---------|--------|-----------------|
| Media | `{diff_stats["media"]:.6f}` | {'≈ 0: distribuzione centrata (nessun bias sistematico)' if abs(diff_stats["media"]) < result.threshold/10 else '≠ 0: presenza di bias direzionale'} |
| Std | `{diff_stats["std"]:.6f}` | Dispersione storica delle differenze |
| Skewness | `{diff_stats["skewness"]:.4f}` | {'≈ 0: simmetrica' if abs(diff_stats["skewness"]) < 0.5 else ('> 0: coda destra più lunga (bias rialzista)' if diff_stats["skewness"] > 0 else '< 0: coda sinistra più lunga (bias ribassista)')} |
| Kurtosis | `{diff_stats["kurtosis_excess"]:.4f}` | {'> 0: code più pesanti del normale (leptokurtic)' if diff_stats["kurtosis_excess"] > 0 else '≈ normale'} |

> 📌 **Validità statistica:** per una distribuzione normale con k=1, ci si aspetterebbe
> il ~32% di osservazioni oltre 1-sigma. Il risultato effettivo è
> **{diff_stats["pct_oltre_soglia"]:.1f}%** — uno scostamento da questa attesa indica
> {'code più pesanti del normale (excess kurtosis positiva).' if diff_stats["pct_oltre_soglia"] > 32 else 'code più leggere del normale.' if diff_stats["pct_oltre_soglia"] < 32 else 'distribuzione vicina alla normalità.'}
""")

fig3 = build_diff_histogram(result, ticker_label)
st.plotly_chart(fig3, width="stretch")

st.divider()


# ============================================================
# SEZIONE 5 — GRAFICO 4: % NON ERGODICI NEL TEMPO
# ============================================================
st.subheader("⏱️ Grafico 4 — Evoluzione Temporale dell'Ergodicità")
st.markdown(f"""
Il grafico mostra la **percentuale rolling** (finestra {rolling_window} giorni) di giorni
non ergodici nel corso del tempo. Picchi elevati corrispondono a periodi di forte
instabilità o cambio di regime.

Valori di riferimento storici per **{ticker_label}**:
- Media storica: **{result.pct_non_ergodic:.1f}%**
- Periodi di crisi nota tendono a mostrare picchi significativamente superiori alla media

> 📌 Questo grafico è utile per identificare **cluster temporali** di non-ergodicità,
> che spesso precedono o accompagnano crisi finanziarie, recessioni o cambi strutturali
> nel mercato.
""")

fig4 = build_rolling_pct_chart(result, ticker_label)
st.plotly_chart(fig4, width="stretch")

st.divider()


# ============================================================
# SEZIONE 6 — GRAFICO 5 + TABELLA PER DECENNIO
# ============================================================
st.subheader("📅 Analisi per Decennio")
st.markdown("""
Questa sezione aggrega i dati per decennio per mostrare **come l'ergodicità del mercato
è cambiata nel corso della storia**. Le barre **rosse** indicano decenni con una percentuale
di giorni non ergodici superiore alla media storica; le barre **verdi** indicano decenni
con maggiore stabilità.

Decenni tipicamente caratterizzati da alta non-ergodicità: anni '70 (stagflazione),
anni 2000 (dot-com crash + crisi finanziaria), anni 2020 (Covid, inflazione).
""")

col_chart, col_table = st.columns([1.2, 1])

with col_chart:
    fig5 = build_decade_bar_chart(decade_stats)
    st.plotly_chart(fig5, width="stretch")

with col_table:
    st.markdown("**Statistiche dettagliate per decennio:**")
    display_stats = decade_stats[[
        "decade", "giorni_totali", "giorni_non_ergodici",
        "pct_non_ergodici", "diff_medio", "diff_std"
    ]].copy()
    display_stats.columns = [
        "Decennio", "Giorni", "Non Ergodici", "% Non Erg.", "Diff Media", "Diff Std"
    ]
    # Colora la colonna "% Non Erg." con gradiente rosso/giallo/verde via CSS puro
    # (senza dipendenza da matplotlib, che background_gradient richiede a runtime)
    avg_pct_decade = display_stats["% Non Erg."].mean()

    def color_pct(val):
        """Restituisce il colore CSS in base alla % rispetto alla media storica."""
        try:
            v = float(str(val).replace("%", ""))
        except ValueError:
            return ""
        if v > avg_pct_decade * 1.3:
            return "background-color: #8B1A1A; color: #FFD0D0"   # rosso scuro
        elif v > avg_pct_decade:
            return "background-color: #B35C00; color: #FFE0B0"   # arancio
        elif v > avg_pct_decade * 0.7:
            return "background-color: #5D7A1A; color: #E0F0B0"   # giallo-verde
        else:
            return "background-color: #1A5C1A; color: #C0F0C0"   # verde

    st.dataframe(
        display_stats.style
        .format({
            "Giorni": "{:,.0f}",
            "Non Ergodici": "{:,.0f}",
            "% Non Erg.": "{:.1f}%",
            "Diff Media": "{:.6f}",
            "Diff Std": "{:.6f}",
        })
        .map(color_pct, subset=["% Non Erg."]),
        width="stretch",
        hide_index=True,
    )

st.divider()


# ============================================================
# SEZIONE 7 — EXPANDER METODOLOGIA E RIFERIMENTI
# ============================================================
with st.expander("🔬 Metodologia, Note Tecniche e Riferimenti"):
    st.markdown(f"""
    ## Metodologia

    ### Dati
    - **Fonte:** EODHD Historical Data API (`eodhd.com`)
    - **Ticker usato:** `{eodhd_ticker}`
    - **Periodo:** `{result.df.index[0].date()}` → `{result.df.index[-1].date()}`
    - **Prezzo di riferimento:** `adjusted_close` (corretto per dividendi e split)
    - **Totale barre:** `{result.n_total:,}` giorni di trading

    ### Calcoli
    - **Log-return:** `r_t = ln(P_t / P_{{t-1}})`
    - **Media temporale (rolling):** `μ_T(t) = mean(r_{{t-N+1}}, ..., r_t)`, N = {rolling_window}
    - **Media spaziale (expanding):** `μ_E(t) = mean(r_1, ..., r_t)` (min {rolling_window} osservazioni)
    - **Differenza:** `Δ(t) = μ_T(t) − μ_E(t)`
    - **Soglia (SEM):** `threshold = k × σ / √N` — Standard Error of the Mean
      - `σ_globale` = `{result.sigma_global:.6f}` (deviazione standard di tutti i log-return)
      - `SEM = σ/√N` = `{result.sigma_global:.6f}` / √{rolling_window} = `{result.sem:.6f}`
      - `k` = `{result.k_mult:.2f}` → soglia = `{result.threshold:.6f}`
    - **Modalità soglia:** `{threshold_mode}` {'(Standard Error of the Mean, raccomandato)' if threshold_mode == 'sem' else '(valore fisso inserito manualmente)'}
    - **Classificazione:** giorno non ergodico ⟺ `|Δ(t)| > {result.threshold:.6f}`

    ### Fondamento Statistico della Soglia SEM

    La rolling mean calcolata su N osservazioni i.i.d. con deviazione standard σ ha
    un'incertezza statistica pari a **σ/√N** (Standard Error of the Mean). Una deviazione
    |Δ| = |rolling − expanding| che supera **k × σ/√N** è statisticamente significativa
    al livello di confidenza corrispondente a k:

    | k | Confidenza | Interpretazione |
    |---|-----------|-----------------|
    | 1.00 | 68.3% | 1-sigma (test permissivo) |
    | 1.75 | ~92%  | **Default — replica ±0.0011 per SPX/252g** |
    | 1.96 | 95.0% | Standard scientifico (2-sigma) |
    | 2.58 | 99.0% | Test molto conservativo |

    Proprietà desiderabili rispetto a soglie basate su `std(Δ)`:
    - ✓ Scala con la volatilità dell'asset (asset più volatili → banda più larga)
    - ✓ Si restringe al crescere di N (finestre più lunghe → stime più precise)
    - ✓ È interpretabile come test t sulla media con N gradi di libertà
    - ✓ Non è influenzata da autocorrelazione della serie Δ stessa

    ### Statistiche della differenza Δ
    | Metrica | Valore |
    |---------|--------|
    | σ globale (log-return) | `{result.sigma_global:.8f}` |
    | SEM = σ/√N | `{result.sem:.8f}` |
    | Soglia = k × SEM | `{result.threshold:.8f}` |
    | Media Δ | `{diff_stats["media"]:.8f}` |
    | Std Δ | `{diff_stats["std"]:.8f}` |
    | Skewness | `{diff_stats["skewness"]:.4f}` |
    | Kurtosis (excess) | `{diff_stats["kurtosis_excess"]:.4f}` |
    | 5° percentile | `{diff_stats["percentile_5"]:.8f}` |
    | 95° percentile | `{diff_stats["percentile_95"]:.8f}` |
    | Min | `{diff_stats["minimo"]:.8f}` |
    | Max | `{diff_stats["massimo"]:.8f}` |

    ### Riferimenti Bibliografici
    1. **Peters, O. (2019).** *The ergodicity problem in economics.* Nature Physics, 15, 1216–1221.
       DOI: 10.1038/s41567-019-0732-0
    2. **Peters, O., & Gell-Mann, M. (2016).** *Evaluating gambles using dynamics.*
       Chaos, 26(2), 023103.
    3. **Taleb, N.N. (2018).** *Skin in the Game.* Random House.
    4. **Mandelbrot, B., & Hudson, R.L. (2004).** *The (Mis)behavior of Markets.*
       Basic Books. — Sul problema delle distribuzioni a code pesanti nei mercati finanziari.

    ### Limitazioni e avvertenze
    - L'analisi è **descrittiva**, non predittiva: la classificazione di un giorno come
      non ergodico descrive una condizione passata, non garantisce alcun segnale operativo.
    - La soglia è sensibile alla lunghezza della finestra rolling e alla modalità di calcolo.
      Testare più configurazioni (es. k=0.5, 1.0, 1.5) per robustezza.
    - Per asset con storia breve (< 5 anni), l'expanding mean potrebbe essere poco stabile
      nelle prime osservazioni.
    """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "📊 **Kriterion Quant** — Finanza Quantitativa | "
    "Dati: EODHD Historical Data | "
    "Questa analisi è fornita a scopo educativo e non costituisce consulenza finanziaria."
)
