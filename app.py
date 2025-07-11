# app.py
import streamlit as st
from rag_utils import load_chroma_db, chat_with_rag
from stock_analysis import get_stock_metrics, explain_metrics
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from Nifty_Prediction import (
    get_nifty_history,
    forecast_next_7,
    get_sentiment_news,
)
from plotly.graph_objects import Candlestick, Figure


# Page configuration (must be first Streamlit command)
st.set_page_config(page_title="RAG Financial Analysis & Stock Dashboard", layout="wide")

# Top navigation tabs (centered)
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
tabs = st.tabs(["Home", "Nifty", "Chat Bot", "Stock Analysis"])
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Home Page
# ----------------------
with tabs[0]:
    st.title("📈 Stock Analysis Dashboard")
    st.markdown(
        "Welcome to the Stock Analysis Dashboard!\n"
        "\n"
        "- Use the **Nifty** tab to explore Nifty-specific analytics.\n"
        "- Head over to **Chat Bot** to interact with the Annual Report RAG system.\n"
        "- Go to **Stock Analysis** for detailed company financials and insights."
    )

# ----------------------
# Nifty Page
# ----------------------

with tabs[1]:
    st.title("📊 Nifty Insights (Candlesticks + Forecast)")

    from Nifty_Prediction import (
        get_nifty_history,
        forecast_next_7,
        get_sentiment_news,
        interpret_index_difference
    )

    # 1) Load 1 year of Nifty history
    hist = get_nifty_history(period_days=365).reset_index()
    df_fcst = forecast_next_7(hist.set_index("Date"))

    # 2) Build candlestick + forecast chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist["Date"],
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Actual",
            ),
            go.Scatter(
                x=df_fcst["Date"],
                y=df_fcst["Close"],
                mode="lines+markers",
                name="7-Day Forecast",
                line=dict(color="orange", dash="dash", width=2),
                marker=dict(size=6)
            ),
        ]
    )

    fig.update_layout(
        title="Nifty 50: 1-Year Daily Candlestick + 7-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3) Top 30-day Nifty news
    st.subheader("📰 Top Nifty News (Last 30 Days)")
    news_items = get_sentiment_news(top_n=5, lookback_days=30)
    if not news_items:
        st.warning("No recent news found.")
    else:
        for item in news_items:
            st.markdown(
                f"""
                <div style="border:1px solid #444; border-left:5px solid {item['color']}; 
                            padding:12px; margin-bottom:12px; border-radius:6px;
                            background:#333; color:white;">
                    <strong>{item['date']}</strong><br>
                    <a href="{item['url']}" target="_blank" style="color:#4EA5F7;">{item['title']}</a><br>
                    <span style="color:{item['color']}; font-weight:bold;">[{item['sentiment']}]</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4) Global Index Comparison Cards
    st.subheader("🌍 Global Indices Comparison (vs Nifty)")
    indices = {
        "Dow Jones (DJI)": "^DJI",
        "S&P 500 (GSPC)": "^GSPC",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
        "DAX": "^GDAXI",
    }

    nifty_close = hist["Close"].iloc[-1]
    cards = st.columns(2)
    for i, (name, ticker) in enumerate(indices.items()):
        hist2 = yf.Ticker(ticker).history(period="2d")["Close"]
        if len(hist2) >= 2:
            prev, last = hist2.iloc[0], hist2.iloc[-1]
            pct = round(((last - prev) / prev) * 100, 2)
            color = "green" if pct >= 0 else "red"

            # LLM interpretation
            diff_pct, interpretation = interpret_index_difference(nifty_close, last, name)
        else:
            last, pct, diff_pct, interpretation = "N/A", "N/A", "N/A", "Data unavailable"
            color = "grey"

        with cards[i % 2]:
            st.markdown(
                f"""
                <div style="background:#444; color:white; border-radius:10px; 
                            padding:14px; margin-bottom:16px; box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
                    <h4 style="margin:0;">{name}</h4>
                    <p style="margin:4px 0;"><strong>Last Close:</strong> ₹{last}</p>
                    <p style="margin:4px 0; color:{color};"><strong>1-Day Change:</strong> {pct}%</p>
                    <p style="margin:6px 0; font-style:italic;">📉 {interpretation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------
# Chat Bot Page
# ----------------------
with tabs[2]:
    st.title("🤖 Annual Report RAG Chatbot")

    uploaded_2022_23 = st.file_uploader("Upload 2022–2023 Annual Report PDF", type="pdf", key="pdf22")
    uploaded_2023_24 = st.file_uploader("Upload 2023–2024 Annual Report PDF", type="pdf", key="pdf23")

    if uploaded_2022_23 and uploaded_2023_24:
        if "db" not in st.session_state:
            with st.spinner("Processing PDFs & building the knowledge base..."):
                db = load_chroma_db(uploaded_2022_23, uploaded_2023_24)
                st.session_state["db"] = db
        else:
            db = st.session_state["db"]
        st.success(f"✅ Vector store ready for {uploaded_2022_23.name} & {uploaded_2023_24.name}")

        st.header("💬 Ask Questions About the Reports")
        user_question = st.text_input("Enter your question:")
        if user_question:
            with st.spinner("Generating answer..."):
                answer = chat_with_rag(db, user_question)
            st.markdown(f"**Answer:** {answer}")
    else:
        st.info("👆 Please upload both annual reports to proceed.")

# ----------------------
# Stock Analysis Page
# ----------------------

with tabs[3]:
    st.title("🔍 Fundamental Stock Analysis")
    st.write("")
    col1, col2 = st.columns([2,1])
    with col1:
        symbol = st.text_input("Enter NSE Symbol (e.g., RELIANCE, TCS)", value="RELIANCE").upper()
    with col2:
        generate = st.button("Generate Analysis")

    if not symbol:
        st.warning("👆 Please enter a valid NSE symbol to proceed.")
    elif not generate:
        st.info("Click 'Generate Analysis' to fetch metrics and insights.")
    else:
        with st.spinner(f"Fetching and explaining metrics for {symbol}..."):
            try:
                df_metrics = get_stock_metrics(symbol)
                df_explained = explain_metrics(df_metrics, symbol)

                if df_explained.empty:
                    st.error(f"No data found for symbol: {symbol}")
                else:
                    st.subheader(f"📊 Metrics & Insights for {symbol}")
                    cols = st.columns(2)
                    for idx, row in enumerate(df_explained.itertuples()):
                        col = cols[idx % 2]
                        with col:
                            card_html = f"""
                            <div style="
                                background-color: #555;
                                color: #fff;
                                border-radius: 12px;
                                padding: 16px;
                                margin: 8px 0;
                                box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
                            ">
                                <h4 style="margin-bottom:8px;">{row.metric}</h4>
                                <p style="margin:4px 0; font-size:14px;"><strong>Value:</strong> {row.value}</p>
                                <p style="margin:8px 0; font-size:14px; line-height:1.5;">{row.explanation}</p>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error fetching or explaining metrics: {e}")
