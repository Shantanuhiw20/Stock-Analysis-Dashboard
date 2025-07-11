# nifty_predict.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os

# Load environment variables
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
llm = ChatGroq(model_name=LLM_MODEL, temperature=0.7)

def get_nifty_history(period_days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    ticker = yf.Ticker("^NSEI")
    hist = ticker.history(
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )
    return hist

def plot_history(hist):
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Nifty 50 Close Price")
    return fig

def forecast_next_7(hist,
                    model_path="bilstm_forecaster.keras",
                    scaler_path="scaler.pkl",
                    lookback=252):
    df_raw = hist.copy()

    # Compute Variance as in training
    raw_var = df_raw['High'] - df_raw['Low']
    df_raw['Variance'] = np.where(
        (df_raw['High'] - df_raw['Close']).abs() < (df_raw['Close'] - df_raw['Low']).abs(),
        raw_var, -raw_var
    )

    df = df_raw.drop(columns=['High','Low','Dividends','Stock Splits'], errors='ignore').round(2)

    # Load model and scaler
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    # Save original stats for denormalization
    orig_means = df.mean()
    orig_stds = df.std()

    # Normalize like training
    df_norm = df.copy()
    df_norm[df.columns] = df_norm[df.columns].subtract(orig_means).div(orig_stds)
    df_norm = df_norm.round(4)

    # Forecast next 7 days
    SEQ_LEN, HORIZON, FEATURES = 26, 7, 4
    last_seq = df_norm[['Open','Close','Volume','Variance']].values[-SEQ_LEN:].reshape(1, SEQ_LEN, FEATURES)
    pred_norm = model.predict(last_seq)
    pred_norm = pred_norm.reshape(HORIZON, FEATURES)

    df_forecast_norm = pd.DataFrame(pred_norm[:, :3], columns=['Open', 'Close', 'Volume'])
    df_forecast = (
        df_forecast_norm.mul(orig_stds[['Open','Close','Volume']], axis=1)
                          .add(orig_means[['Open','Close','Volume']], axis=1)
                          .round(2)
    )

    # Build forecast dataframe
    last_date = pd.to_datetime(df.index[-1])
    future_dates = [last_date + timedelta(days=i+1) for i in range(HORIZON)]
    df_forecast.insert(0, 'Date', pd.to_datetime(future_dates))
    return df_forecast

def get_sentiment_news(top_n=5, lookback_days=15):
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    end_date = datetime.now()
    cutoff   = end_date - timedelta(days=lookback_days)

    ticker = yf.Ticker("^NSEI")
    news   = ticker.news or []

    filtered = []
    for item in news:
        ts = item.get("providerPublishTime")
        if ts is None:
            continue

        published = datetime.fromtimestamp(ts)
        if published < cutoff:
            continue

        title = item.get("title", "")
        score = sia.polarity_scores(title)["compound"]
        if   score >= 0.05:  sentiment, color = "Positive", "green"
        elif score <= -0.05: sentiment, color = "Negative", "red"
        else:                sentiment, color = "Neutral",  "grey"

        filtered.append({
            "date":      published.strftime("%Y-%m-%d"),
            "title":     title,
            "url":       item.get("link", "#"),
            "sentiment": sentiment,
            "color":     color
        })

        if len(filtered) >= top_n:
            break

    return filtered

def interpret_index_difference(nifty_close, other_close, name):
    diff_pct = round(((nifty_close - other_close) / other_close) * 100, 2)
    prompt = (
        f"Nifty closed at ₹{nifty_close} and {name} closed at ₹{other_close}. "
        f"Explain what is {name} in one line"
        f"Begin directly with insight—no introductory phrases or role mentions. "
        f"State if it's positive, neutral, or a warning, and why. "
        "Limit each explanation to around 30 words."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    interpretation = response.content.strip()
    return diff_pct, interpretation
