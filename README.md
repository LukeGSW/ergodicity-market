# 📊 Ergodicità del Mercato — Rolling vs Global Mean

**Kriterion Quant** | Finanza Quantitativa Educativa

Dashboard Streamlit per lo studio dell'**ergodicità dei mercati finanziari**:
confronto tra media temporale (rolling) e media spaziale (expanding) dei rendimenti
logaritmici giornalieri, con classificazione automatica dei giorni non ergodici.

---

## 🔬 Cosa fa l'app

1. Scarica dati storici OHLCV da **EODHD** per qualsiasi asset (indici, azioni, ETF, crypto, futures)
2. Calcola i **rendimenti logaritmici** giornalieri
3. Confronta la **media temporale** (rolling, finestra configurabile) con la **media spaziale** (expanding)
4. Classifica ogni giorno come *ergodico* o *non ergodico* in base alla soglia configurata
5. Produce 5 grafici interattivi + tabella per decennio + statistiche complete

---

## 📂 Struttura Repository

```
ergodicity-market/
├── app.py                    # Entry point Streamlit
├── requirements.txt          # Dipendenze Python
├── .streamlit/
│   └── config.toml           # Tema dark + configurazione server
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py       # Fetch EODHD con caching (ttl=1h)
│   ├── calculations.py       # Logica ergodicità (rolling/expanding/soglia)
│   └── charts.py             # 5 grafici Plotly dark theme
├── .gitignore
└── README.md
```

---

## 🚀 Deploy su Streamlit Cloud

### 1. Fork / Push su GitHub

```bash
git init
git add .
git commit -m "feat: ergodicity market dashboard"
git remote add origin https://github.com/TUO-USERNAME/ergodicity-market.git
git push -u origin main
```

### 2. Crea l'app su Streamlit Cloud

1. Vai su [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
2. Seleziona il repository GitHub
3. Imposta `app.py` come file principale
4. In **Advanced settings → Secrets** incolla:

```toml
EODHD_API_KEY = "la-tua-chiave-eodhd"
```

> Ottieni una chiave gratuita su [eodhd.com](https://eodhd.com) (piano free: 20 anni di dati EOD per indici US)

5. Clicca **Deploy** → l'app sarà live su `https://[nome].streamlit.app`

---

## 💻 Run Locale

```bash
# 1. Clona il repository
git clone https://github.com/TUO-USERNAME/ergodicity-market.git
cd ergodicity-market

# 2. Crea e attiva virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# .\venv\Scripts\activate       # Windows

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura API key locale
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
EODHD_API_KEY = "la-tua-chiave-eodhd"
EOF

# 5. Avvia l'app
streamlit run app.py
```

---

## 🗺️ Ticker supportati (EODHD)

| Shortcut | Ticker EODHD | Asset |
|----------|-------------|-------|
| SPX | `GSPC.INDX` | S&P 500 |
| NDX | `NDX.INDX` | Nasdaq 100 |
| DAX | `GDAXI.INDX` | DAX 40 |
| FTMIB | `FTSEMIB.INDX` | FTSE MIB |
| BTCUSD | `BTC-USD.CC` | Bitcoin |
| ETHUSD | `ETH-USD.CC` | Ethereum |
| NI225 | `N225.INDX` | Nikkei 225 |
| HSI | `HSI.INDX` | Hang Seng |
| GC1 | `GC.COMM` | Gold Futures |
| CL1 | `CL.COMM` | Crude Oil WTI |
| VIX | `VIX.INDX` | CBOE VIX |

Sono supportati anche **ticker diretti** nel formato EODHD (es. `AAPL.US`, `ENI.MI`, `SPY.US`).

---

## 📚 Riferimenti Teorici

- **Peters, O. (2019).** *The ergodicity problem in economics.* Nature Physics, 15, 1216–1221.
- **Peters, O., & Gell-Mann, M. (2016).** *Evaluating gambles using dynamics.* Chaos, 26(2).
- **Taleb, N.N. (2018).** *Skin in the Game.* Random House.

---

## ⚠️ Disclaimer

Questa applicazione è fornita **a solo scopo educativo e di ricerca quantitativa**.
Non costituisce consulenza finanziaria. I risultati non sono raccomandazioni di investimento.

---

*Kriterion Quant — Finanza Quantitativa | [kriterionquant.com](https://kriterionquant.com)*
