import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title and description
st.title("Company Ranking Dashboard")
st.write("This dashboard provides insights into the ranked companies based on financial metrics.")

# Add expandable section for KPI definitions
with st.expander("**Key Performance Indicators (KPIs) and Definitions**"):
    st.write("""
    Below are the definitions and formulas for each KPI used in the ranking algorithm:

    1. **Revenue Growth (3-year CAGR)** [revenue_growth]  
       - **Definition**: Average annual revenue growth rate over 3 years.  
       - **Formula**: ((Revenue_Latest / Revenue_Oldest)^(1/3)) - 1  

    2. **EPS Growth (3-year CAGR)** [eps_growth]  
       - **Definition**: Average annual earnings per share growth rate over 3 years.  
       - **Formula**: ((EPS_Latest / EPS_Oldest)^(1/3)) - 1  

    3. **Free Cash Flow (latest year)** [free_cash_flow]  
       - **Definition**: Cash from operations available for reinvestment, dividends, or debt reduction.  
       - **Formula**: Operating Cash Flow - Capital Expenditures  

    4. **Free Cash Flow Per Share (FCFPS)** [fcfps]  
       - **Definition**: Represents the amount of free cash flow available to each shareholder.  
       - **Formula**: Free Cash Flow / Shares Outstanding  

    5. **Free Cash Flow Yield** [free_cash_flow_yield]  
       - **Definition**: Free cash flow relative to market capitalization or stock price.  
       - **Formula**: Free Cash Flow / Market Capitalization  

    6. **Return on Equity (ROE)** [roe]  
       - **Definition**: Profitability relative to shareholders' equity.  
       - **Formula**: Net Income / Shareholder Equity  

    7. **PE Ratio** [pe_ratio]  
       - **Definition**: Shows how the market values the company's future prospects.  
       - **Formula**: Current Stock Price / Earnings Per Share  

    8. **Current Annual Growth (CAG)** [cag]  
       - **Definition**: Measures the most recent year-over-year growth in earnings per share.  
       - **Formula**: (EPS_This_Year / EPS_Last_Year) - 1  

    9. **PEG Ratio** [peg_ratio]  
       - **Definition**: Compares PE ratio to future growth rate.  
       - **Formula**: PE Ratio / (Current Annual Growth * 100)  

    10. **Relative Strength (vs. S&P 500)** [relative_strength]  
       - **Definition**: Stock's price performance relative to a benchmark index.  
       - **Formula**: (Stock Return / Benchmark Return)  
    """)

# Load the data from file upload or use the default file
uploaded_file = st.file_uploader("Upload a CSV file (or leave blank to use default)", type="csv")

@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.success("Data loaded successfully from uploaded file.")
            return df
        except Exception as e:
            st.error(f"Error loading the file: {e}")
            return None  # Or handle the error more gracefully
    else:
        try:
            df = pd.read_csv("ranked_stocks_median.csv")  # Default file
            st.info("Using the default dataset (ranked_stocks_median.csv).")
            return df
        except FileNotFoundError:
            st.error("Default dataset 'ranked_stocks_median.csv' not found.  Please upload a CSV file.")
            return None
        except Exception as e:
            st.error(f"Error loading default file: {e}")
            return None


df = load_data(uploaded_file)

if df is not None:  # Only proceed if the DataFrame is loaded
    # Display top 10 ranked companies
    st.header("Top 10 Ranked Companies")
    st.write("Below are the top 10 companies based on their weighted scores:")
    st.dataframe(df.head(10))

    # Weighted Score Distribution
    st.header("Weighted Score Distribution")
    st.write("The distribution of weighted scores across all companies:")
    fig, ax = plt.subplots()
    sns.histplot(df["weighted_score"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Weighted Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Variable Contributions
    st.header("Variable Contributions")
    st.write("The contribution of each variable to the overall weighted score:")

    # Calculate the average normalized score for each variable
    variable_columns = [col for col in df.columns if "_normalized" in col]
    variable_contributions = df[variable_columns].mean().reset_index()
    variable_contributions.columns = ["Variable", "Average Score"]

    # Plot the contributions
    fig, ax = plt.subplots()
    sns.barplot(x="Average Score", y="Variable", data=variable_contributions, ax=ax)
    ax.set_xlabel("Average Normalized Score")
    ax.set_ylabel("Variable")
    st.pyplot(fig)

    # Filter by Rank
    st.header("Filter Companies by Rank")
    min_rank = st.slider("Minimum Rank", 1, len(df), 1)
    max_rank = st.slider("Maximum Rank", 1, len(df), 100)
    filtered_df = df[(df["rank"] >= min_rank) & (df["rank"] <= max_rank)]
    st.write(f"Displaying companies ranked between {min_rank} and {max_rank}:")
    st.dataframe(filtered_df)

    # Search by Company
    st.header("Search by Company")
    search_term = st.text_input("Enter a company name or ticker:")
    if search_term:
        search_results = df[df["stock"].str.contains(search_term, case=False)]
        if not search_results.empty:
            st.write("Search Results:")
            st.dataframe(search_results)
        else:
            st.write("No matching companies found.")

    # Download Results
    st.header("Download Results")
    st.write("Click the button below to download the full ranked list as a CSV file.")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ranked_companies.csv",
        mime="text/csv",
    )
