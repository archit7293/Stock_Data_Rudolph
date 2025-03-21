import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import json
import time

def pull_stock_data(progress_bar, status_text):
    # STEP 1: Fetch filtered stock data
    api_key = 'QolfHdg3kmWuMxSEMFEZ8J9pbTpl48uC'
    screener_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=1000000000&volumeMoreThan=750000&exchange=NASDAQ,NYSE&limit=3000&&apikey={api_key}'
    
    status_text.text("Fetching filtered stock data...")
    response = requests.get(screener_url)
    filtered_stocks = response.json()
    df = pd.DataFrame(filtered_stocks)
    df.to_csv("FMP_fitered_stock_data.csv", index=False)
    status_text.text("Filtered stock data saved to CSV.")
    progress_bar.progress(10)  # Update progress bar

    # STEP 2: Fetch KPIs for each stock
    stock_symbols = df['symbol']
    all_kpis = {}
    chunk_size = 50
    sleep_time = 30
    symbol_sleep_time = 2
    max_retries = 3

    total_stocks = len(stock_symbols)
    processed_stocks = 0

    for chunk_index, i in enumerate(range(0, total_stocks, chunk_size), start=1):
        chunk = stock_symbols[i:i + chunk_size]
        status_text.text(f"Processing chunk {chunk_index} of {len(stock_symbols) // chunk_size + 1}...")
        
        for symbol in chunk:
            status_text.text(f"Fetching KPIs for {symbol}...")
            retries = 0
            while retries < max_retries:
                try:
                    kpis = get_stock_kpis(symbol)
                    if kpis:
                        all_kpis[symbol] = kpis
                    else:
                        status_text.text(f"No data returned for {symbol}")
                    break
                except Exception as e:
                    error_message = str(e).lower()
                    status_text.text(f"Error processing {symbol}: {e}")
                    if "rate limit" in error_message or "too many requests" in error_message:
                        wait_time = 60 * (retries + 1)
                        status_text.text(f"Rate limit encountered. Sleeping for {wait_time} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        status_text.text(f"Non-rate limit error for {symbol}. Skipping.")
                        break
            else:
                status_text.text(f"Max retries exceeded for {symbol}. Skipping.")
            time.sleep(symbol_sleep_time)
            processed_stocks += 1
            progress_bar.progress(10 + int((processed_stocks / total_stocks) * 80)  # Update progress bar

        status_text.text(f"Processed chunk {chunk_index}. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    # Save KPIs to JSON
    with open("stock_kpis.json", "w") as f:
        json.dump(all_kpis, f, indent=4)
    status_text.text("KPIs saved to JSON.")
    progress_bar.progress(95)  # Update progress bar

    # STEP 3: Convert JSON to DataFrame
    with open("stock_kpis.json", "r") as f:
        stock_kpis_data = json.load(f)
    stock_kpis_df = pd.DataFrame(stock_kpis_data).T.reset_index().rename(columns={"index": "stock"})
    stock_kpis_df.to_csv('stock_kpis.csv', index=False)
    status_text.text("KPIs converted to DataFrame and saved to CSV.")
    progress_bar.progress(100)  # Update progress bar

    # STEP 4: Ranking Algorithm
    df = pd.read_csv('stock_kpis.csv')
    growth_columns = ["revenue_growth", "eps_growth", "cag"]
    absolute_columns = ["free_cash_flow", "fcfps", "free_cash_flow_yield", "roe", "pe_ratio", "peg_ratio", "relative_strength"]

    for col in growth_columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in absolute_columns:
        df[col].fillna(df[col].median(), inplace=True)

    weights = {
        "revenue_growth": 0.20,
        "eps_growth": 0.20,
        "free_cash_flow": 0.15,
        "free_cash_flow_yield": 0.20,
        "roe": 0.10,
        "peg_ratio": 0.10,
        "relative_strength": 0.05,
    }

    for column in weights.keys():
        df[column + "_normalized"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    df["weighted_score"] = (
        df["revenue_growth_normalized"] * weights["revenue_growth"]
        + df["eps_growth_normalized"] * weights["eps_growth"]
        + df["free_cash_flow_normalized"] * weights["free_cash_flow"]
        + df["free_cash_flow_yield_normalized"] * weights["free_cash_flow_yield"]
        + df["roe_normalized"] * weights["roe"]
        + df["peg_ratio_normalized"] * weights["peg_ratio"]
        + df["relative_strength_normalized"] * weights["relative_strength"]
    )

    df["rank"] = df["weighted_score"].rank(ascending=False)
    df = df.sort_values(by="rank")
    df.to_csv("ranked_stocks_median.csv", index=False)
    status_text.text("Ranking complete. Final CSV saved.")
    return "ranked_stocks_median.csv"

def main():
    st.title("Stock Data Puller")
    st.write("Click the button below to start pulling stock data.")

    if st.button('START DATA PULL'):
        progress_bar = st.progress(0)  # Initialize progress bar
        status_text = st.empty()  # Placeholder for status updates

        with st.spinner('Pulling data... This may take a while.'):
            output_file = pull_stock_data(progress_bar, status_text)
            st.success('Data pull complete!')

            # Provide a download button for the CSV file
            with open(output_file, "rb") as file:
                btn = st.download_button(
                    label="Download ranked_stocks_median.csv",
                    data=file,
                    file_name="ranked_stocks_median.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()
