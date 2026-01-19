"""
Data Loading Utilities
======================
Functions for loading and validating stock data and PCA results.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to data directory - update this to match your GitHub repo structure
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# GitHub raw file URLs - Your FinTech repository
GITHUB_BASE_URL = "https://raw.githubusercontent.com/DrOsmanDatascience/FinTech/main/data"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_stock_data(use_sample=True):
    """
    Load stock information and factor data.
    
    Parameters:
    -----------
    use_sample : bool
        If True, generates sample data for demonstration.
        Set to False and update file paths to use real data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing stock information with columns:
        - ticker: Stock ticker symbol
        - permno: PERMNO identifier
        - company_name: Full company name
        - sector: Industry sector
        - market_weight: Portfolio weight (%)
        - value_score, quality_score, fin_strength_score, 
          momentum_score, volatility_score, liquidity_score: Factor scores
    """
    if use_sample:
        return _generate_sample_stock_data()
    
    # ----- REAL DATA LOADING -----
    # Uncomment and modify the path below to load from your GitHub repo or local files
    
    # Option 1: Load from local file
    # file_path = os.path.join(DATA_DIR, 'stock_data.csv')
    # return pd.read_csv(file_path)
    
    # Option 2: Load from GitHub
    # url = f"{GITHUB_BASE_URL}/stock_data.csv"
    # return pd.read_csv(url)
    
    raise FileNotFoundError("Real data loading not configured. Set use_sample=True or update paths.")


def load_pca_data(use_sample=True):
    """
    Load PCA results including scores and cluster assignments.
    
    Parameters:
    -----------
    use_sample : bool
        If True, generates sample data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing PCA results with columns:
        - ticker: Stock ticker symbol
        - permno: PERMNO identifier
        - company_name: Full company name
        - pc1: Principal Component 1 score
        - pc2: Principal Component 2 score
        - cluster: Cluster assignment (0-3)
        - quadrant: Quadrant label
        - Plus all columns from stock_data
    """
    if use_sample:
        return _generate_sample_pca_data()
    
    # ----- REAL DATA LOADING -----
    # Uncomment and modify to load from your data sources
    
    # Option 1: Load from local file
    # file_path = os.path.join(DATA_DIR, 'pca_results.csv')
    # return pd.read_csv(file_path)
    
    # Option 2: Load from GitHub
    # url = f"{GITHUB_BASE_URL}/pca_results.csv"
    # return pd.read_csv(url)
    
    raise FileNotFoundError("Real data loading not configured.")


def load_time_series_pca(ticker, use_sample=True):
    """
    Load historical PCA positions for a specific stock.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    use_sample : bool
        If True, generates sample time series data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, ticker, pc1, pc2
    """
    if use_sample:
        return _generate_sample_time_series(ticker)
    
    # ----- REAL DATA LOADING -----
    # file_path = os.path.join(DATA_DIR, 'time_series_pca.csv')
    # df = pd.read_csv(file_path)
    # return df[df['ticker'] == ticker]
    
    raise FileNotFoundError("Real data loading not configured.")


def validate_ticker(user_input, stock_data, input_type="ticker"):
    """
    Validate user input and return stock information if valid.
    
    Parameters:
    -----------
    user_input : str
        Ticker symbol or PERMNO entered by user
    stock_data : pd.DataFrame
        Stock data to search
    input_type : str
        Either "ticker" or "permno"
    
    Returns:
    --------
    tuple (bool, dict or str)
        (True, stock_info_dict) if valid
        (False, error_message) if invalid
    """
    if not user_input:
        return False, "Please enter a value"
    
    if input_type == "ticker":
        ticker = user_input.upper().strip()
        match = stock_data[stock_data['ticker'] == ticker]
    else:
        try:
            permno = int(user_input)
            match = stock_data[stock_data['permno'] == permno]
        except ValueError:
            return False, "PERMNO must be a number"
    
    if len(match) == 0:
        return False, f"No stock found for {input_type}: {user_input}"
    
    # Return first match as dictionary
    stock_info = match.iloc[0].to_dict()
    
    # Add quadrant name if not present
    if 'quadrant_name' not in stock_info:
        stock_info['quadrant_name'] = _get_quadrant_name(
            stock_info.get('pc1', 0), 
            stock_info.get('pc2', 0)
        )
    
    return True, stock_info


def _get_quadrant_name(pc1, pc2):
    """Determine quadrant name based on PC scores."""
    if pc1 >= 0 and pc2 >= 0:
        return "High Quality / Large-Liquid"
    elif pc1 < 0 and pc2 >= 0:
        return "Lower Quality / Large-Liquid"
    elif pc1 >= 0 and pc2 < 0:
        return "High Quality / Cash-Rich"
    else:
        return "Lower Quality / Cash-Rich"


# =============================================================================
# SAMPLE DATA GENERATORS
# =============================================================================

def _generate_sample_stock_data():
    """Generate realistic sample stock data for demonstration."""
    
    # Sample S&P 500 stocks across different quadrants
    stocks = [
        # High Quality / Large-Liquid (Quadrant: top-right)
        {'ticker': 'AAPL', 'permno': 14593, 'company_name': 'Apple Inc', 'sector': 'Technology'},
        {'ticker': 'MSFT', 'permno': 10107, 'company_name': 'Microsoft Corp', 'sector': 'Technology'},
        {'ticker': 'JNJ', 'permno': 22111, 'company_name': 'Johnson & Johnson', 'sector': 'Healthcare'},
        {'ticker': 'PG', 'permno': 18163, 'company_name': 'Procter & Gamble', 'sector': 'Consumer Staples'},
        {'ticker': 'V', 'permno': 93436, 'company_name': 'Visa Inc', 'sector': 'Financials'},
        {'ticker': 'MA', 'permno': 89393, 'company_name': 'Mastercard Inc', 'sector': 'Financials'},
        {'ticker': 'HD', 'permno': 66181, 'company_name': 'Home Depot', 'sector': 'Consumer Discretionary'},
        {'ticker': 'CL', 'permno': 19174, 'company_name': 'Colgate-Palmolive', 'sector': 'Consumer Staples'},
        
        # High Quality / Cash-Rich (Quadrant: bottom-right)
        {'ticker': 'GOOGL', 'permno': 90319, 'company_name': 'Alphabet Inc', 'sector': 'Technology'},
        {'ticker': 'BRK.B', 'permno': 83443, 'company_name': 'Berkshire Hathaway', 'sector': 'Financials'},
        {'ticker': 'COST', 'permno': 75100, 'company_name': 'Costco Wholesale', 'sector': 'Consumer Staples'},
        {'ticker': 'LLY', 'permno': 24840, 'company_name': 'Eli Lilly', 'sector': 'Healthcare'},
        {'ticker': 'ABBV', 'permno': 91233, 'company_name': 'AbbVie Inc', 'sector': 'Healthcare'},
        {'ticker': 'MRK', 'permno': 22752, 'company_name': 'Merck & Co', 'sector': 'Healthcare'},
        
        # Lower Quality / Large-Liquid (Quadrant: top-left)
        {'ticker': 'TSLA', 'permno': 93436, 'company_name': 'Tesla Inc', 'sector': 'Consumer Discretionary'},
        {'ticker': 'NVDA', 'permno': 93436, 'company_name': 'NVIDIA Corp', 'sector': 'Technology'},
        {'ticker': 'AMD', 'permno': 47896, 'company_name': 'Advanced Micro Devices', 'sector': 'Technology'},
        {'ticker': 'NFLX', 'permno': 89368, 'company_name': 'Netflix Inc', 'sector': 'Communication Services'},
        {'ticker': 'CRM', 'permno': 89531, 'company_name': 'Salesforce Inc', 'sector': 'Technology'},
        
        # Lower Quality / Smaller (Quadrant: bottom-left)
        {'ticker': 'RIVN', 'permno': 94111, 'company_name': 'Rivian Automotive', 'sector': 'Consumer Discretionary'},
        {'ticker': 'LCID', 'permno': 94222, 'company_name': 'Lucid Group', 'sector': 'Consumer Discretionary'},
        {'ticker': 'SNAP', 'permno': 92111, 'company_name': 'Snap Inc', 'sector': 'Communication Services'},
        {'ticker': 'COIN', 'permno': 94333, 'company_name': 'Coinbase Global', 'sector': 'Financials'},
        {'ticker': 'PLTR', 'permno': 93888, 'company_name': 'Palantir Technologies', 'sector': 'Technology'},
    ]
    
    np.random.seed(42)
    
    for i, stock in enumerate(stocks):
        # Generate factor scores based on realistic sector characteristics
        sector = stock['sector']
        
        if stock['ticker'] in ['AAPL', 'MSFT', 'JNJ', 'PG', 'V', 'MA', 'HD', 'CL']:
            # High quality, large/liquid
            stock['value_score'] = np.random.uniform(0.3, 0.7)
            stock['quality_score'] = np.random.uniform(0.7, 0.95)
            stock['fin_strength_score'] = np.random.uniform(0.6, 0.9)
            stock['momentum_score'] = np.random.uniform(0.4, 0.7)
            stock['volatility_score'] = np.random.uniform(0.2, 0.4)
            stock['liquidity_score'] = np.random.uniform(0.7, 0.95)
            stock['market_weight'] = np.random.uniform(1.5, 7.0)
        elif stock['ticker'] in ['GOOGL', 'BRK.B', 'COST', 'LLY', 'ABBV', 'MRK']:
            # High quality, cash-rich
            stock['value_score'] = np.random.uniform(0.4, 0.8)
            stock['quality_score'] = np.random.uniform(0.65, 0.9)
            stock['fin_strength_score'] = np.random.uniform(0.7, 0.95)
            stock['momentum_score'] = np.random.uniform(0.5, 0.8)
            stock['volatility_score'] = np.random.uniform(0.25, 0.45)
            stock['liquidity_score'] = np.random.uniform(0.4, 0.65)
            stock['market_weight'] = np.random.uniform(0.8, 3.5)
        elif stock['ticker'] in ['TSLA', 'NVDA', 'AMD', 'NFLX', 'CRM']:
            # Lower quality, large/liquid (high growth)
            stock['value_score'] = np.random.uniform(0.1, 0.35)
            stock['quality_score'] = np.random.uniform(0.35, 0.55)
            stock['fin_strength_score'] = np.random.uniform(0.4, 0.65)
            stock['momentum_score'] = np.random.uniform(0.6, 0.95)
            stock['volatility_score'] = np.random.uniform(0.6, 0.85)
            stock['liquidity_score'] = np.random.uniform(0.75, 0.95)
            stock['market_weight'] = np.random.uniform(0.5, 4.0)
        else:
            # Lower quality, smaller/volatile
            stock['value_score'] = np.random.uniform(0.05, 0.25)
            stock['quality_score'] = np.random.uniform(0.15, 0.4)
            stock['fin_strength_score'] = np.random.uniform(0.2, 0.45)
            stock['momentum_score'] = np.random.uniform(0.3, 0.7)
            stock['volatility_score'] = np.random.uniform(0.7, 0.95)
            stock['liquidity_score'] = np.random.uniform(0.3, 0.55)
            stock['market_weight'] = np.random.uniform(0.05, 0.4)
    
    return pd.DataFrame(stocks)


def _generate_sample_pca_data():
    """Generate sample PCA results based on stock data."""
    
    stock_data = _generate_sample_stock_data()
    
    np.random.seed(42)
    
    # Calculate PC scores based on factor loadings from the document
    # PC1 loadings: ROA (0.44), ROE (0.36), Earnings Yield (0.39), Cash/Debt (0.36), Momentum (0.26)
    #               Volatility (-0.34), Book-to-Market (-0.35)
    # PC2 loadings: ADDV (0.54), Debt-to-Assets (0.48)
    #               Cash/Debt (-0.34), Sales-to-Price (-0.29), Gross Profitability (-0.27)
    
    pca_data = stock_data.copy()
    
    # Calculate PC1 (Quality/Stability)
    pca_data['pc1'] = (
        0.44 * pca_data['quality_score'] +  # ROA proxy
        0.36 * pca_data['quality_score'] +  # ROE proxy
        0.39 * pca_data['value_score'] +    # Earnings Yield proxy
        0.36 * pca_data['fin_strength_score'] +  # Cash/Debt
        0.26 * pca_data['momentum_score'] -
        0.34 * pca_data['volatility_score'] -
        0.35 * (1 - pca_data['value_score'])  # Book-to-Market (inverse of value)
    )
    
    # Calculate PC2 (Size/Leverage)
    pca_data['pc2'] = (
        0.54 * pca_data['liquidity_score'] +  # ADDV
        0.48 * (1 - pca_data['fin_strength_score']) -  # Debt-to-Assets
        0.34 * pca_data['fin_strength_score'] -  # Cash/Debt
        0.29 * pca_data['value_score'] -  # Sales-to-Price
        0.27 * pca_data['quality_score']  # Gross Profitability
    )
    
    # Normalize to realistic ranges
    pca_data['pc1'] = (pca_data['pc1'] - pca_data['pc1'].mean()) / pca_data['pc1'].std() * 2.5
    pca_data['pc2'] = (pca_data['pc2'] - pca_data['pc2'].mean()) / pca_data['pc2'].std() * 1.5
    
    # Add some noise
    pca_data['pc1'] += np.random.normal(0, 0.3, len(pca_data))
    pca_data['pc2'] += np.random.normal(0, 0.2, len(pca_data))
    
    # Assign clusters based on k-means style clustering (0-3)
    def assign_cluster(row):
        if row['pc1'] >= 0 and row['pc2'] >= 0:
            return 0  # High quality, large/liquid
        elif row['pc1'] < 0 and row['pc2'] >= 0:
            return 1  # Lower quality, large/liquid
        elif row['pc1'] >= 0 and row['pc2'] < 0:
            return 2  # High quality, cash-rich
        else:
            return 3  # Lower quality, smaller
    
    pca_data['cluster'] = pca_data.apply(assign_cluster, axis=1)
    pca_data['quadrant_name'] = pca_data.apply(
        lambda r: _get_quadrant_name(r['pc1'], r['pc2']), axis=1
    )
    
    # Add percentile rankings within quadrant
    for cluster in pca_data['cluster'].unique():
        mask = pca_data['cluster'] == cluster
        cluster_data = pca_data[mask]
        
        for factor in ['value_score', 'quality_score', 'fin_strength_score', 
                       'momentum_score', 'volatility_score', 'liquidity_score']:
            pctl_col = factor.replace('_score', '_pctl')
            pca_data.loc[mask, pctl_col] = cluster_data[factor].rank(pct=True) * 100
    
    return pca_data


def _generate_sample_time_series(ticker, periods=24):
    """
    Generate sample historical PCA positions for animation.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker to generate history for
    periods : int
        Number of monthly periods to generate
    
    Returns:
    --------
    pd.DataFrame
        Historical PCA positions with columns: date, ticker, pc1, pc2
    """
    pca_data = _generate_sample_pca_data()
    
    # Get current position
    stock = pca_data[pca_data['ticker'] == ticker]
    if len(stock) == 0:
        # Return empty DataFrame if ticker not found
        return pd.DataFrame(columns=['date', 'ticker', 'pc1', 'pc2'])
    
    current_pc1 = stock['pc1'].values[0]
    current_pc2 = stock['pc2'].values[0]
    
    np.random.seed(hash(ticker) % (2**32))
    
    # Generate historical positions with mean reversion toward current
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='M')
    
    # Start from a different position and drift toward current
    start_pc1 = current_pc1 + np.random.uniform(-2, 2)
    start_pc2 = current_pc2 + np.random.uniform(-1, 1)
    
    pc1_values = np.linspace(start_pc1, current_pc1, periods) + np.random.normal(0, 0.15, periods)
    pc2_values = np.linspace(start_pc2, current_pc2, periods) + np.random.normal(0, 0.1, periods)
    
    return pd.DataFrame({
        'date': dates,
        'ticker': ticker,
        'pc1': pc1_values,
        'pc2': pc2_values
    })


# =============================================================================
# DATA VALIDATION & PREPROCESSING
# =============================================================================

def preprocess_data(df):
    """
    Preprocess loaded data to ensure consistent format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data from file
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed data
    """
    # Convert ticker to uppercase
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].str.upper().str.strip()
    
    # Ensure numeric columns are numeric
    numeric_cols = ['pc1', 'pc2', 'market_weight', 'permno',
                    'value_score', 'quality_score', 'fin_strength_score',
                    'momentum_score', 'volatility_score', 'liquidity_score']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df = df.fillna({
        'market_weight': 0,
        'value_score': 0.5,
        'quality_score': 0.5,
        'fin_strength_score': 0.5,
        'momentum_score': 0.5,
        'volatility_score': 0.5,
        'liquidity_score': 0.5
    })
    
    return df


if __name__ == "__main__":
    # Test data generation
    print("Testing data loader...")
    
    stock_data = load_stock_data(use_sample=True)
    print(f"\nStock data shape: {stock_data.shape}")
    print(stock_data.head())
    
    pca_data = load_pca_data(use_sample=True)
    print(f"\nPCA data shape: {pca_data.shape}")
    print(pca_data[['ticker', 'company_name', 'pc1', 'pc2', 'cluster']].head(10))
    
    # Test validation
    is_valid, result = validate_ticker("AAPL", pca_data, "ticker")
    print(f"\nValidation test (AAPL): {is_valid}")
    if is_valid:
        print(f"Result: {result['ticker']} - {result['company_name']}")
    
    # Test time series
    ts_data = load_time_series_pca("CL", use_sample=True)
    print(f"\nTime series data shape: {ts_data.shape}")
    print(ts_data.head())
