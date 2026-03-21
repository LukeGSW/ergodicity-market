"""
data_fetcher.py — Fetch e caching dati storici da EODHD.

Tutte le funzioni di fetch usano @st.cache_data per evitare chiamate API
ridondanti durante le sessioni Streamlit. Il TTL di default è 3600s (1 ora).

Fonte dati: EODHD Historical Data (https://eodhd.com)
Formato ticker EODHD:
  Indici:  GSPC.INDX (S&P 500), NDX.INDX, GDAXI.INDX, VIX.INDX ...
  Azioni:  AAPL.US, ENI.MI, SAP.XETRA ...
  ETF:     SPY.US, QQQ.US ...
  Futures: GC.COMM (gold), CL.COMM (crude oil) ...
  Crypto:  BTC-USD.CC, ETH-USD.CC ...
"""

from __future__ import annotations

import requests
import pandas as pd
import streamlit as st


# ============================================================
# MAPPA SCORCIATOIE → TICKER EODHD
# ============================================================

TICKER_MAP: dict[str, tuple[str, str]] = {
    # (eodhd_ticker, nome_leggibile)
    "SPX":    ("GSPC.INDX",   "S&P 500"),
    "NDX":    ("NDX.INDX",    "Nasdaq 100"),
    "DAX":    ("GDAXI.INDX",  "DAX 40"),
    "FTMIB":  ("FTSEMIB.INDX","FTSE MIB"),
    "BTCUSD": ("BTC-USD.CC",  "Bitcoin / USD"),
    "ETHUSD": ("ETH-USD.CC",  "Ethereum / USD"),
    "NI225":  ("N225.INDX",   "Nikkei 225"),
    "HSI":    ("HSI.INDX",    "Hang Seng Index"),
    "GC1":    ("GC.COMM",     "Gold Futures"),
    "CL1":    ("CL.COMM",     "Crude Oil WTI"),
    "VIX":    ("VIX.INDX",    "CBOE VIX"),
}


# ============================================================
# FETCH DATI EODHD
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(
    ticker: str,
    end: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Scarica l'intera storia disponibile da EODHD per il ticker richiesto.

    Il parametro `from` non viene passato all'API per evitare il troncamento:
    EODHD restituisce al massimo ~19.000 record per chiamata. Se si parte dal 1950
    con un asset longevo (es. SPX), il limite viene raggiunto e i dati più recenti
    vengono tagliati. Scaricando senza from_date e filtrando lato Python si ottiene
    sempre l'ultima candela disponibile.

    Args:
        ticker:  Simbolo nel formato EODHD (es. 'GSPC.INDX', 'SPY.US', 'BTC-USD.CC')
        end:     Data fine 'YYYY-MM-DD' — esplicita nella firma perché
                 entra nella chiave della cache e la invalida ogni giorno.
        api_key: Chiave API EODHD (da st.secrets)

    Returns:
        DataFrame con indice DatetimeIndex e colonne:
        open, high, low, close, volume, adjusted_close (float64)

    Raises:
        requests.HTTPError: se la risposta API non è 2xx
        ValueError:         se nessun dato è disponibile per il ticker
    """
    url = (
        f"https://eodhd.com/api/eod/{ticker}"
        f"?to={end}"
        f"&period=d&api_token={api_key}&fmt=json"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if not data:
        raise ValueError(
            f"Nessun dato disponibile per '{ticker}'. "
            "Verifica il ticker."
        )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    wanted_cols = ["open", "high", "low", "close", "volume", "adjusted_close"]
    available = [c for c in wanted_cols if c in df.columns]
    return df[available].astype(float)


def resolve_ticker(raw_input: str) -> tuple[str, str]:
    """
    Converte l'input utente nel ticker EODHD corretto e nel nome leggibile.

    Supporta sia scorciatoie (es. 'SPX') sia ticker EODHD diretti (es. 'GSPC.INDX').

    Args:
        raw_input: Stringa inserita dall'utente o selezionata dalla sidebar

    Returns:
        Tuple (eodhd_ticker, nome_leggibile)
    """
    key = raw_input.strip().upper()
    if key in TICKER_MAP:
        return TICKER_MAP[key]
    # Input diretto in formato EODHD: restituisce così com'è
    return (raw_input.strip(), raw_input.strip())
