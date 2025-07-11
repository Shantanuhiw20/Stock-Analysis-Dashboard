# stock_analysis.py
import yfinance as yf
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.config import LLM_MODEL


def get_stock_metrics(symbol: str) -> pd.DataFrame:
    """
    Fetches key financial metrics for the given Indian stock symbol (without .NS).
    Returns a DataFrame with metric, value, and placeholder for explanation.
    """
    ticker = yf.Ticker(f"{symbol}.NS")
    info = ticker.info

    # Key metrics
    metrics = {
        "Market Cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price/Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Return on Equity (ROE)": info.get("returnOnEquity"),
        "Return on Assets (ROA)": info.get("returnOnAssets"),
        "Debt/Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "Profit Margin": info.get("profitMargins"),
        "EBITDA Margin": info.get("ebitdaMargins"),
    }

    df = pd.DataFrame([
        {"metric": k, "value": (None if v is None else round(v, 2)), "explanation": ""}
        for k, v in metrics.items()
    ])
    return df


def explain_metrics(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Uses LLM to explain each metric in the DataFrame for long-term investors.
    Updates 'explanation' column with concise analysis (max ~200 words).
    """
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["metric", "value", "symbol"],
        template=(
            "Explain the significance of the {metric} value of {value} for {symbol} in clear, investor-friendly language. "
            "Begin directly with insight—no introductory phrases or role mentions. "
            "State if it's positive, neutral, or a warning, and why. "
            "Limit each explanation to around 200 words."
        )
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    for idx, row in df.iterrows():
        val = row["value"]
        if val is None:
            df.at[idx, "explanation"] = "Data not available"
            continue
        response = chain.run({
            "metric": row["metric"],
            "value": val,
            "symbol": symbol
        })
        df.at[idx, "explanation"] = response.strip()

    return df


if __name__ == "__main__":
    symbol = input("Enter NSE symbol (e.g., RELIANCE): ").upper()
    df = get_stock_metrics(symbol)
    df = explain_metrics(df, symbol)
    pd.set_option("display.max_colwidth", None)
    print(df.to_string(index=False))
