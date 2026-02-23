import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ── Página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Bot Backtester",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Trading Bot Backtester")
st.caption("Estrategia Close-to-Close & Overnight Agresiva")

# ── Sidebar — Configuración ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    ticker = st.selectbox("Acción", ["GOOGL", "AAPL", "NVDA", "META", "AMZN", "MSFT"])
    start  = st.date_input("Fecha inicio", value=pd.to_datetime("2020-01-01"))
    end    = st.date_input("Fecha fin",    value=pd.to_datetime("2024-12-31"))
    run    = st.button("▶️  Correr Backtest", use_container_width=True)

# ── Lógica ───────────────────────────────────────────────────
def load_data(ticker, start, end):
    df = yf.download(ticker, start=str(start), end=str(end),
                     auto_adjust=True, progress=False)
    df = df[["Open","High","Low","Close"]].copy()
    df.columns = ["open","high","low","close"]
    df["prev_close"] = df["close"].shift(1)
    return df.dropna()

def strategy_a(df):
    trades, position = [], None
    for date, row in df.iterrows():
        if position is None:
            if row["open"] < row["prev_close"]:
                position = {"buy_date": date, "buy_price": row["open"]}
        else:
            if row["close"] > row["prev_close"]:
                pnl = row["close"] - position["buy_price"]
                trades.append({
                    "Estrategia": "A - Close-to-Close",
                    "Entrada": position["buy_date"].date(),
                    "Precio compra": round(position["buy_price"], 2),
                    "Salida": date.date(),
                    "Precio venta": round(row["close"], 2),
                    "P&L $": round(pnl, 2),
                    "Días": (date - position["buy_date"]).days,
                    "Resultado": "✅ WIN" if pnl > 0 else "❌ LOSS",
                })
                position = None
    if position:
        last, last_date = df.iloc[-1], df.index[-1]
        pnl = last["close"] - position["buy_price"]
        trades.append({
            "Estrategia": "A - Close-to-Close",
            "Entrada": position["buy_date"].date(),
            "Precio compra": round(position["buy_price"], 2),
            "Salida": last_date.date(),
            "Precio venta": round(last["close"], 2),
            "P&L $": round(pnl, 2),
            "Días": (last_date - position["buy_date"]).days,
            "Resultado": "✅ WIN (forzado)" if pnl > 0 else "❌ LOSS (forzado)",
        })
    return pd.DataFrame(trades)

def strategy_b(df):
    trades = []
    for i in range(len(df) - 1):
        today = df.iloc[i]
        nxt   = df.iloc[i + 1]
        if today["open"] < today["prev_close"]:
            if nxt["open"] > today["close"]:
                pnl = nxt["open"] - today["open"]
                trades.append({
                    "Estrategia": "B - Overnight",
                    "Entrada": df.index[i].date(),
                    "Precio compra": round(today["open"], 2),
                    "Salida": df.index[i + 1].date(),
                    "Precio venta": round(nxt["open"], 2),
                    "P&L $": round(pnl, 2),
                    "Días": 1,
                    "Resultado": "✅ WIN" if pnl > 0 else "❌ LOSS",
                })
    return pd.DataFrame(trades)

def metrics(t, name, initial_price):
    if t.empty:
        return None
    wins   = t[t["P&L $"] > 0]
    losses = t[t["P&L $"] <= 0]
    gp     = wins["P&L $"].sum() if not wins.empty else 0
    gl     = abs(losses["P&L $"].sum()) if not losses.empty else 0.0001
    eq     = t["P&L $"].cumsum()
    dd     = eq - eq.cummax()
    retorno_pct = (t["P&L $"].sum() / initial_price) * 100
    return {
        "Estrategia": name,
        "Operaciones": len(t),
        "Rendimiento %": f"{retorno_pct:.1f}%",
        "P&L Total": f"${t['P&L $'].sum():,.2f}",
        "Ganancia prom": f"${wins['P&L $'].mean():.2f}" if not wins.empty else "-",
        "Pérdida prom":  f"${losses['P&L $'].mean():.2f}" if not losses.empty else "-",
        "Profit Factor": f"{gp/gl:.2f}",
        "Mejor trade": f"${t['P&L $'].max():.2f}",
        "Peor trade":  f"${t['P&L $'].min():.2f}",
        "Max Drawdown": f"${dd.min():.2f}",
        "Días prom":   f"{t['Días'].mean():.1f}",
    }

def equity_chart(ta, tb, df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Precio
    axes[0].plot(df.index, df["close"], color="#4fc3f7", linewidth=1.2)
    axes[0].set_title(f"Precio cierre — {ticker}")
    axes[0].set_ylabel("USD")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Equity A
    if not ta.empty:
        ta_dates = pd.to_datetime(ta["Salida"])
        eq_a = ta["P&L $"].cumsum()
        axes[1].plot(ta_dates, eq_a, color="#66bb6a", linewidth=1.5)
        axes[1].fill_between(ta_dates, eq_a, 0, where=(eq_a>=0), alpha=0.2, color="green")
        axes[1].fill_between(ta_dates, eq_a, 0, where=(eq_a<0),  alpha=0.2, color="red")
        axes[1].axhline(0, color="#555", linewidth=0.8, linestyle="--")
    axes[1].set_title("Equity Curve — Estrategia A (Close-to-Close)")
    axes[1].set_ylabel("P&L acumulado $")

    # Equity B
    if not tb.empty:
        tb_dates = pd.to_datetime(tb["Salida"])
        eq_b = tb["P&L $"].cumsum()
        axes[2].plot(tb_dates, eq_b, color="#ffa726", linewidth=1.5)
        axes[2].fill_between(tb_dates, eq_b, 0, where=(eq_b>=0), alpha=0.2, color="green")
        axes[2].fill_between(tb_dates, eq_b, 0, where=(eq_b<0),  alpha=0.2, color="red")
        axes[2].axhline(0, color="#555", linewidth=0.8, linestyle="--")
    axes[2].set_title("Equity Curve — Estrategia B (Overnight)")
    axes[2].set_ylabel("P&L acumulado $")

    plt.tight_layout()
    return fig

# ── Ejecución ─────────────────────────────────────────────────
if run:
    with st.spinner(f"Descargando datos de {ticker}..."):
        df = load_data(ticker, start, end)

    st.success(f"✅ {len(df)} días de datos cargados")

    ta = strategy_a(df)
    tb = strategy_b(df)
    initial_price = df["close"].iloc[0]
    ma = metrics(ta, "A - Close-to-Close", initial_price)
    mb = metrics(tb, "B - Overnight", initial_price)

    # Buy & Hold
    bh_pnl = df["close"].iloc[-1] - df["close"].iloc[0]
    bh_pct = bh_pnl / df["close"].iloc[0] * 100

    # ── Métricas superiores ──
    st.subheader("📊 Resumen")
    col1, col2, col3 = st.columns(3)
    col1.metric("Estrategia A — P&L", f"${ta['P&L $'].sum():,.2f}" if not ta.empty else "-")
    col2.metric("Estrategia B — P&L", f"${tb['P&L $'].sum():,.2f}" if not tb.empty else "-")
    col3.metric(f"Buy & Hold {ticker}", f"${bh_pnl:,.2f}", f"{bh_pct:.1f}%")

    # ── Tabla de métricas ──
    st.subheader("📋 Métricas detalladas")
    rows = [m for m in [ma, mb] if m]
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Estrategia"), use_container_width=True)

    # ── Gráfica ──
    st.subheader("📈 Equity Curves")
    fig = equity_chart(ta, tb, df)
    st.pyplot(fig)

    # ── Tabs con log de trades ──
    st.subheader("📄 Log de Operaciones")
    tab1, tab2 = st.tabs(["Estrategia A", "Estrategia B"])
    with tab1:
        if not ta.empty:
            st.dataframe(ta, use_container_width=True)
            csv_a = ta.to_csv(index=False).encode()
            st.download_button("⬇️ Descargar CSV — Estrategia A", csv_a, "trades_A.csv", "text/csv")
        else:
            st.info("Sin operaciones generadas.")
    with tab2:
        if not tb.empty:
            st.dataframe(tb, use_container_width=True)
            csv_b = tb.to_csv(index=False).encode()
            st.download_button("⬇️ Descargar CSV — Estrategia B", csv_b, "trades_B.csv", "text/csv")
        else:
            st.info("Sin operaciones generadas.")

else:
    st.info("👈  Configura los parámetros en el panel izquierdo y presiona **Correr Backtest**.")
    st.markdown("""
    ### Cómo funciona
    **Estrategia A — Close-to-Close**
    - Compra 1 acción si el precio de apertura es menor al cierre anterior
    - Vende al cierre cuando el cierre supera el cierre anterior
    - Mantiene posición hasta que se cumpla la condición

    **Estrategia B — Overnight Agresiva**
    - Compra 1 acción si el precio de apertura es menor al cierre anterior
    - Vende a la apertura del día siguiente si abre por encima del cierre anterior
    - Operaciones de máximo 1 día
    """)
