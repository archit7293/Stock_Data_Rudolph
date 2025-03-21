import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import json
import time
from io import BytesIO

# --- Streamlit App Setup ---
st.title("Stock Data and KPI Analysis")

# --- Input API Key ---
api_key = st.text_input("Enter your FMP API Key:", type="password")

# --- STEP 1 (Data Filtering) ---
@st.cache_data
def get_filtered_stock_data(api_key):
    if not api_key:
        return None, "Please enter your API key."

    screener_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=1000000000&volumeMoreThan=750000&exchange=NASDAQ,NYSE&limit=3000&&apikey={api_key}'
    try:
        response = requests.get(screener_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        filtered_stocks = response.json()
        df = pd.DataFrame(filtered_stocks)
        return df, None  # Return DataFrame and no error message
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data from FMP: {e}"
    except json.JSONDecodeError:
        return None, "Error decoding JSON response.  Check your API key and the FMP API status."


# --- STEP 2 (KPI Calculation) ---
def get_stock_kpis(stock_symbol, benchmark_symbol="^GSPC"):
    """Fetches financial data and calculates KPIs for a given stock symbol.

    Args:
        stock_symbol (str): The stock symbol (e.g., "AAPL").
        benchmark_symbol (str, optional): The benchmark symbol (e.g., "^GSPC" for S&P 500). Defaults to "^GSPC".

    Returns:
        dict: A dictionary containing the calculated KPIs, or None if an error occurs.
    """

    try:
        stock = yf.Ticker(stock_symbol)
        benchmark = yf.Ticker(benchmark_symbol)

        # Fetch financial statements (income statement, cash flow)
        income_stmt = stock.income_stmt.T  # Transpose for easier date-based indexing
        cash_flow = stock.cashflow.T       # Transpose for easier date-based indexing
        balance_sheet = stock.balance_sheet.T

        # Fetch stock prices (sorted by date ascending)
        stock_prices = stock.history(period="1y").sort_index(ascending=True)
        benchmark_prices = benchmark.history(period="1y").sort_index(ascending=True)

        # Shares outstanding and current price
        shares_outstanding = stock.info.get('sharesOutstanding', 0)
        current_price = stock.info.get('currentPrice', 0)
        market_cap = current_price * shares_outstanding

        # --------------------------------------------
        # Calculate KPIs (with error handling)
        # --------------------------------------------

        # 1. Revenue Growth (3-year CAGR)
        try:
            # Extract the last 4 years of revenue data (to calculate 3-year CAGR)
            revenue = income_stmt['Total Revenue'].dropna()
            if len(revenue) >= 4:
                rev_latest = revenue.iloc[0]  # Latest year (first row)
                rev_oldest = revenue.iloc[3]  # 4th year back (3-year gap)
                revenue_growth = (rev_latest / rev_oldest) ** (1/3) - 1
            else:
                revenue_growth = float('nan')
        except KeyError:
            revenue_growth = float('nan')

        # 2. EPS Growth (3-year CAGR)
        try:
            net_income = income_stmt['Net Income'].dropna()
            if len(net_income) >= 4:
                eps_latest = net_income.iloc[0] / shares_outstanding  # Latest year
                eps_oldest = net_income.iloc[3] / shares_outstanding  # 4th year back
                eps_growth = (eps_latest / eps_oldest) ** (1/3) - 1
            else:
                eps_growth = float('nan')
        except KeyError:
            eps_growth = float('nan')

        # 3. Free Cash Flow (most recent year)
        try:
            free_cash_flow = cash_flow['Free Cash Flow'].iloc[0]
        except KeyError:
            free_cash_flow = float('nan')
        # Calculate Free Cash Flow Per Share (FCFPS)
        fcfps = free_cash_flow / shares_outstanding if shares_outstanding != 0 else float('nan')

        # 4. Free Cash Flow Yield
        free_cash_flow_yield = free_cash_flow / market_cap if (market_cap != 0 and not pd.isna(free_cash_flow)) else float('nan')

        # 5. Return on Equity (ROE)
        try:
            net_income_latest = income_stmt['Net Income'].iloc[0]
            equity_latest = balance_sheet['Stockholders Equity'].iloc[0]
            roe = net_income_latest / equity_latest if equity_latest != 0 else float('nan')
        except KeyError:
            roe = float('nan')

        # # 6. PEG Ratio
        # pe_ratio = stock.info.get('trailingPE', float('nan'))
        # # if pd.isna(eps_growth) or eps_growth <= 0:
        # #     peg_ratio = float('nan')
        # # else:
        # peg_ratio = pe_ratio / (eps_growth * 100)  # Convert Ebita growth to percentage

        # 7. Relative Strength (1-year performance vs. benchmark)
        try:
            stock_return = (stock_prices['Close'].iloc[-1] / stock_prices['Close'].iloc[0]) - 1
            benchmark_return = (benchmark_prices['Close'].iloc[-1] / benchmark_prices['Close'].iloc[0]) - 1
            relative_strength = stock_return / benchmark_return if benchmark_return != 0 else float('nan')
        except IndexError:
            relative_strength = float('nan')

        # 8. ebitda Growth (3-year CAGR)

        try:
            ebitda = income_stmt['EBITDA'].dropna()
            if len(net_income) >= 4:
                e_latest = ebitda.iloc[0]  # Latest year
                e_oldest = ebitda.iloc[3]  # 4th year back
                e_growth = (e_latest / e_oldest) ** (1/3) - 1 #(3 year gap)
            else:
                e_growth = float('nan')
        except KeyError:
            e_growth = float('nan')


        # 9. annual growth rate
        # Current annual growth rate is 2024 Earnings per share (EPS) divided
        # by 2023 EPS. Sometimes number will be negative.
        try:
            net_income = income_stmt['Net Income'].dropna()
            shares_outstanding = balance_sheet['Ordinary Shares Number'].dropna()  # Shares outstanding for each year
            net_income = income_stmt['Net Income'].dropna()
            eps_0 = net_income.iloc[0] / shares_outstanding.iloc[0]  # Latest year
            eps_1 = net_income.iloc[1] / shares_outstanding.iloc[1]  # 1 year back
            cag = (eps_0 / eps_1) ** (1/1) - 1 #(1 year gap)
        except KeyError:
            cag = float('nan')

        # 6. PEG Ratio
        pe_ratio = stock.info.get('trailingPE', float('nan'))
        # if pd.isna(eps_growth) or eps_growth <= 0:
        #     peg_ratio = float('nan')
        # else:
        peg_ratio = pe_ratio / (cag * 100)  # Convert cag to percentage


        # Print results
        print(f"Revenue Growth (3-year CAGR): {revenue_growth:.2%}")
        print(f"EPS Growth (3-year CAGR): {eps_growth:.2%}")
        print(f"Free Cash Flow (latest year): ${free_cash_flow / 1e9:.2f}B")
        print(f"Free Cash Flow Per Share (FCFPS): ${fcfps:.2f}")
        print(f"Free Cash Flow Yield: {free_cash_flow_yield:.2%}")
        print(f"ROE: {roe:.2%}")
        print(f"PE Ratio: {pe_ratio:.2f}")
        #print(f"Ebitda Growth (3-year CAGR): {e_growth:.2%}")
        print(f"Current Annual Growth: {cag:.2%}")
        print(f"PEG Ratio: {peg_ratio:.2f}")
        print(f"Relative Strength (vs. S&P 500): {relative_strength:.2f}")
        # Store results in a dictionary
        kpis = {
            "revenue_growth": revenue_growth,
            "eps_growth": eps_growth,
            "free_cash_flow": free_cash_flow,
            "fcfps": fcfps,
            "free_cash_flow_yield": free_cash_flow_yield,
            "roe": roe,
            "pe_ratio": pe_ratio,
            "cag": cag,
            "peg_ratio": peg_ratio,
            "relative_strength": relative_strength
        }

        return kpis

    except Exception as e:
        print(f"Error processing {stock_symbol}: {e}")
        return None


def process_data(api_key):
    """Main function to fetch data, calculate KPIs, and save the results."""
    # --- STEP 1: Filter Stock Data ---
    df, error_message = get_filtered_stock_data(api_key)

    if error_message:
        st.error(error_message)
        return None

    if df is None or df.empty:
        st.warning("No stocks found matching the filter criteria.")
        return None

    stock_symbols = df['symbol']
    total_stocks = len(stock_symbols)
    all_kpis = {}
    chunk_size = 50
    sleep_time = 30  # Wait (seconds) between chunks
    symbol_sleep_time = 2  # Wait (seconds) between symbols
    max_retries = 3
    processed_stocks_count = 0 # Track processed stocks

    # Load existing data if resuming from a previous run
    try:
        with open("stock_kpis_partial.json", "r") as f:
            all_kpis = json.load(f)
        processed_symbols = set(all_kpis.keys())
        processed_stocks_count = len(processed_symbols)  #Update count if resuming
    except FileNotFoundError:
        processed_symbols = set()


    total_chunks = (len(stock_symbols) + chunk_size - 1) // chunk_size

    progress_bar = st.progress(0)  # Initialize progress bar
    status_text = st.empty()  # For displaying status messages

    # Set initial progress (STEP 1)
    progress_bar.progress(10)

    for chunk_index, i in enumerate(range(0, len(stock_symbols), chunk_size), start=1):
        chunk = stock_symbols[i:i + chunk_size]
        for symbol in chunk:
            if symbol in processed_symbols:
                status_text.text(f"Skipping already processed symbol: {symbol}")
                continue

            status_text.text(f"Processing {symbol}...")
            retries = 0
            while retries < max_retries:
                try:
                    kpis = get_stock_kpis(symbol)
                    if kpis:
                        all_kpis[symbol] = kpis
                    else:
                        status_text.text(f"No data returned for {symbol}")
                    break  # Exit the retry loop if successful
                except Exception as e:
                    error_message = str(e).lower()
                    status_text.text(f"Error processing {symbol}: {e}")
                    if "rate limit" in error_message or "too many requests" in error_message:
                        wait_time = 60 * (retries + 1)  # Exponential backoff
                        status_text.text(f"Rate limit encountered. Sleeping for {wait_time} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        # For other types of errors, you may choose to skip or handle differently
                        status_text.text(f"Non-rate limit error for {symbol}. Skipping.")
                        break
            else:
                status_text.text(f"Max retries exceeded for {symbol}. Skipping.")
            time.sleep(symbol_sleep_time)  # Wait between symbols
            processed_stocks_count += 1 # Increment counter

            # Update progress bar (STEP 2)
            progress = 10 + int((processed_stocks_count / total_stocks) * 80)
            progress_bar.progress(progress)


        status_text.text(f"Processed chunk {chunk_index} of {total_chunks}")
        # Save the results after each chunk
        with open("stock_kpis_partial.json", "w") as f:
            json.dump(all_kpis, f, indent=4)

        status_text.text(f"Saved progress after chunk {chunk_index}")
        time.sleep(sleep_time)  # Wait before processing the next chunk


    # Save the final results to a JSON file
    with open("stock_kpis.json", "w") as f:
        json.dump(all_kpis, f, indent=4)

    # Update progress bar (STEP 3 - JSON to DataFrame)
    progress_bar.progress(90)
    status_text.text("Converting JSON to DataFrame...")


    # --- STEP 3: JSON to DataFrame and CSV ---
    try:
        with open("stock_kpis.json", "r") as f:
            stock_kpis_data = json.load(f)

        stock_kpis_df = pd.DataFrame(stock_kpis_data).T
        stock_kpis_df = stock_kpis_df.reset_index()
        stock_kpis_df = stock_kpis_df.rename(columns={"index": "stock"})
        stock_kpis_df.to_csv('stock_kpis.csv', index=False)
    except Exception as e:
        st.error(f"Error processing JSON to DataFrame: {e}")
        return None

    # Update progress bar (STEP 4 - Ranking)
    progress_bar.progress(92)
    status_text.text("Ranking stocks...")

    # --- STEP 4: Ranking Algo ---
    try:
        df = pd.read_csv('stock_kpis.csv')

        # Replace NA with median for growth-related metrics
        growth_columns = [
            "revenue_growth",  # 3-year revenue growth
            "eps_growth",      # 3-year EPS growth
            "cag",            # Current Annual Growth
        ]
        for col in growth_columns:
            df[col].fillna(df[col].median(), inplace=True)

        # Replace NA with median for absolute metrics
        absolute_columns = [
            "free_cash_flow",          # Free Cash Flow (latest year)
            "fcfps",                  # Free Cash Flow Per Share
            "free_cash_flow_yield",   # Free Cash Flow Yield
            "roe",                    # Return on Equity
            "pe_ratio",               # PE Ratio
            "peg_ratio",              # PEG Ratio
            "relative_strength",      # Relative Strength
        ]
        for col in absolute_columns:
            df[col].fillna(df[col].median(), inplace=True)

        # Weights for each variable
        weights = {
            "revenue_growth": 0.20,
            "eps_growth": 0.20,
            "free_cash_flow": 0.15,
            "free_cash_flow_yield": 0.20,
            "roe": 0.10,
            "peg_ratio": 0.10,
            "relative_strength": 0.05,
        }

        # Normalize the data (scale to 0-1)
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())

        # Apply normalization to each variable
        for column in weights.keys():
            df[column + "_normalized"] = normalize(df[column])

        # Calculate weighted score
        df["weighted_score"] = (
            df["revenue_growth_normalized"] * weights["revenue_growth"]
            + df["eps_growth_normalized"] * weights["eps_growth"]
            + df["free_cash_flow_normalized"] * weights["free_cash_flow"]
            + df["free_cash_flow_yield_normalized"] * weights["free_cash_flow_yield"]
            + df["roe_normalized"] * weights["roe"]
            + df["peg_ratio_normalized"] * weights["peg_ratio"]
            + df["relative_strength_normalized"] * weights["relative_strength"]
        )

        # Rank companies based on weighted score
        df["rank"] = df["weighted_score"].rank(ascending=False)

        # Sort by rank
        df = df.sort_values(by="rank")

        df.to_csv("ranked_stocks_median.csv", index=False)
    except Exception as e:
        st.error(f"Error during ranking: {e}")
        return None

    # Update progress bar and download (STEP 5)
    progress_bar.progress(95)
    status_text.text("Saving and downloading final results...")
    with open("ranked_stocks_median.csv", "r") as f:
        ranked_data = f.read()

    progress_bar.progress(100)
    st.success("Analysis complete!")
    st.download_button(
        label="Download Ranked Stocks (ranked_stocks_median.csv)",
        data=ranked_data,
        file_name="ranked_stocks_median.csv",
        mime="text/csv"
    )

    return "ranked_stocks_median.csv"  # Return the filename for download


# --- START BUTTON and DATA PROCESSING ---
if st.button("Start Data Pull & Analysis"):
    if not api_key:
        st.error("Please enter your FMP API key.")
    else:
        with st.spinner("Fetching data and calculating KPIs... This may take a while."):
            download_filename = process_data(api_key)
