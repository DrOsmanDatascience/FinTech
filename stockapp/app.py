"""
Stock PCA Cluster Analysis - Streamlit Web Application

A comprehensive interactive dashboard for analyzing stock clusters using
Principal Component Analysis (PCA). Features include:
- Interactive 2D/3D PCA visualizations
- Quadrant-based peer comparison
- Factor breakdown analysis
- Time-lapse animations
- AI-powered chatbot for contextual questions

Author: Beautiful Mind FinTech
Date: 2024
"""

import streamlit as st
import pandas as pd
import os
from typing import Optional

# Import project modules
from config import (
    PAGE_CONFIG,
    GITHUB_DATA_URL,
    FEATURE_COLUMNS,
    FACTOR_CATEGORIES,
    QUADRANTS,
    PC1_INTERPRETATION,
    PC2_INTERPRETATION,
    OPENAI_API_KEY_PLACEHOLDER
)
from utils import (
    load_data,
    preprocess_data,
    get_available_tickers,
    validate_stock_input,
    filter_stock_data,
    compute_pca_and_clusters,
    get_pca_loadings,
    determine_quadrant,
    get_stocks_in_same_quadrant,
    compute_percentile_ranks,
    get_cluster_summary,
    prepare_time_series_data,
    get_factor_breakdown
)
from visualizations import (
    create_pca_scatter_plot,
    create_quadrant_comparison_plot,
    create_factor_radar_chart,
    create_percentile_chart,
    create_timelapse_animation,
    create_3d_pca_plot,
    create_cluster_summary_plot
)
from chatbot import create_chatbot, SAMPLE_QUESTIONS


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(**PAGE_CONFIG)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    /* Quadrant indicator */
    .quadrant-box {
        background-color: #e8f4e8;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'data_loaded': False,
        'raw_data': None,
        'processed_data': None,
        'pca_df': None,
        'pca_model': None,
        'kmeans_model': None,
        'scaler': None,
        'selected_stock': None,
        'selected_stock_data': None,
        'chatbot': None,
        'chat_history': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=True)
def load_and_process_data():
    """Load and preprocess data with caching."""
    try:
        # Load data from GitHub
        raw_data = load_data(use_github=True)
        
        # Preprocess data
        processed_data = preprocess_data(raw_data)
        
        # Compute PCA and clustering
        pca_df, pca_model, kmeans_model, scaler = compute_pca_and_clusters(processed_data)
        
        return raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, None
        
    except Exception as e:
        return None, None, None, None, None, None, str(e)


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with stock selection and controls."""
    
    st.sidebar.markdown("## 📊 Stock Selection")
    
    # Stock input
    st.sidebar.markdown("""
    <div class="info-box">
        Enter a stock ticker (e.g., AAPL, MSFT) or PERMNO to analyze.
    </div>
    """, unsafe_allow_html=True)
    
    stock_input = st.sidebar.text_input(
        "Enter Stock Ticker or PERMNO:",
        placeholder="e.g., AAPL or 14593",
        key="stock_input"
    )
    
    # Validation and selection
    if stock_input and st.session_state.processed_data is not None:
        is_valid, input_type, normalized_value = validate_stock_input(
            st.session_state.processed_data, 
            stock_input
        )
        
        if is_valid:
            st.sidebar.success(f"✅ Found: {normalized_value} ({input_type})")
            st.session_state.selected_stock = {
                'value': normalized_value,
                'type': input_type
            }
        else:
            st.sidebar.error(f"❌ '{stock_input}' not found in dataset")
            st.session_state.selected_stock = None
    
    # Quick selection dropdown (top stocks)
    if st.session_state.pca_df is not None and 'ticker' in st.session_state.pca_df.columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Select")
        
        tickers = [''] + sorted(st.session_state.pca_df['ticker'].dropna().unique().tolist())
        selected_dropdown = st.sidebar.selectbox(
            "Or choose from list:",
            options=tickers,
            key="ticker_dropdown"
        )
        
        if selected_dropdown:
            st.session_state.selected_stock = {
                'value': selected_dropdown,
                'type': 'ticker'
            }
    
    # Display axis interpretations
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Axis Interpretations")
    
    with st.sidebar.expander("PC1 (X-axis): Quality/Stability"):
        st.markdown(f"""
        **Explains ~{PC1_INTERPRETATION['variance_explained']}% of variance**
        
        **High values (→ Right):**
        - {', '.join(PC1_INTERPRETATION['high_meaning'])}
        
        **Low values (← Left):**
        - {', '.join(PC1_INTERPRETATION['low_meaning'])}
        """)
    
    with st.sidebar.expander("PC2 (Y-axis): Size/Leverage"):
        st.markdown(f"""
        **Explains ~{PC2_INTERPRETATION['variance_explained']}% of variance**
        
        **High values (↑ Up):**
        - {', '.join(PC2_INTERPRETATION['high_meaning'])}
        
        **Low values (↓ Down):**
        - {', '.join(PC2_INTERPRETATION['low_meaning'])}
        """)
    
    # OpenAI API Key input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Chatbot Settings")
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key (for chatbot):",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable the AI chatbot feature"
    )
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        st.session_state.chatbot = create_chatbot(api_key)


# =============================================================================
# MAIN CONTENT COMPONENTS
# =============================================================================

def render_main_header():
    """Render the main page header."""
    st.markdown("""
    <div class="main-header">
        📈 Stock PCA Cluster Analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Analyze stocks using Principal Component Analysis (PCA) to understand their 
    characteristics across quality, stability, leverage, and size dimensions.
    """)


def render_stock_overview(stock_data: pd.DataFrame, pca_row: pd.Series):
    """Render the stock overview section."""
    
    ticker = pca_row.get('ticker', 'N/A')
    permno = pca_row.get('permno', 'N/A')
    cluster = pca_row.get('cluster', 'N/A')
    pc1 = pca_row.get('PC1', 0)
    pc2 = pca_row.get('PC2', 0)
    quadrant = determine_quadrant(pc1, pc2)
    quadrant_info = QUADRANTS.get(quadrant, {})
    
    st.markdown(f"## 📊 Analysis: {ticker}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", ticker)
    with col2:
        st.metric("PERMNO", permno)
    with col3:
        st.metric("Cluster", f"Cluster {cluster}")
    with col4:
        st.metric("Quadrant", quadrant)
    
    # Quadrant description
    st.markdown(f"""
    <div class="quadrant-box">
        <h4>{quadrant}: {quadrant_info.get('name', 'Unknown')}</h4>
        <p>{quadrant_info.get('description', '')}</p>
        <p><strong>Characteristics:</strong> {', '.join(quadrant_info.get('characteristics', []))}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PCA scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "PC1 Score (Quality/Stability)", 
            f"{pc1:.3f}",
            delta="Higher Quality" if pc1 >= 0 else "Riskier",
            delta_color="normal" if pc1 >= 0 else "inverse"
        )
    with col2:
        st.metric(
            "PC2 Score (Size/Leverage)", 
            f"{pc2:.3f}",
            delta="Large/Leveraged" if pc2 >= 0 else "Cash-Rich",
            delta_color="off"
        )
    
    return ticker, permno, cluster, pc1, pc2, quadrant


def render_visualizations(
    pca_df: pd.DataFrame,
    selected_ticker: str,
    pca_row: pd.Series,
    quadrant_peers: pd.DataFrame,
    raw_data: pd.DataFrame,
    pca_model,
    scaler
):
    """Render the visualization tabs."""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 PCA Cluster Plot",
        "👥 Quadrant Peers",
        "📊 Factor Analysis",
        "🕐 Time-Lapse",
        "🌐 3D View"
    ])
    
    with tab1:
        st.markdown("### PCA Cluster Visualization")
        st.markdown("""
        This plot shows all stocks positioned based on their quality/stability (PC1) 
        and size/leverage (PC2) characteristics. Your selected stock is highlighted with a ⭐.
        """)
        
        fig = create_pca_scatter_plot(pca_df, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Quadrant Peer Comparison")
        st.markdown(f"""
        Showing {len(quadrant_peers)} stocks that share the same quadrant as {selected_ticker}.
        These are potential peers for comparison.
        """)
        
        if not quadrant_peers.empty:
            fig = create_quadrant_comparison_plot(pca_df, selected_ticker, quadrant_peers)
            st.plotly_chart(fig, use_container_width=True)
            
            # Peer table
            with st.expander("📋 View Peer Table"):
                display_cols = ['ticker', 'permno', 'PC1', 'PC2', 'cluster']
                display_cols = [c for c in display_cols if c in quadrant_peers.columns]
                st.dataframe(quadrant_peers[display_cols].round(3))
        else:
            st.info("No other stocks found in this quadrant.")
    
    with tab3:
        st.markdown("### Factor Breakdown Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            factor_data = get_factor_breakdown(pca_row)
            fig_radar = create_factor_radar_chart(factor_data, selected_ticker)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            # Percentile rankings
            available_features = [c for c in FEATURE_COLUMNS if c in pca_row.index]
            percentiles = compute_percentile_ranks(quadrant_peers, pca_row, available_features)
            
            if percentiles:
                fig_percentile = create_percentile_chart(percentiles, selected_ticker)
                st.plotly_chart(fig_percentile, use_container_width=True)
        
        # Factor details table
        with st.expander("📋 Detailed Factor Values"):
            factor_table = []
            for category, features in FACTOR_CATEGORIES.items():
                for feature in features:
                    if feature in pca_row.index:
                        pct = percentiles.get(feature, 'N/A')
                        factor_table.append({
                            'Category': category,
                            'Factor': feature,
                            'Value': pca_row[feature],
                            'Percentile': f"{pct:.1f}%" if isinstance(pct, (int, float)) else pct
                        })
            
            if factor_table:
                st.dataframe(pd.DataFrame(factor_table))
    
    with tab4:
        st.markdown("### Historical Movement Animation")
        st.markdown("""
        Watch how the stock's position has changed over time in the PCA space.
        Click **Play** to start the animation.
        """)
        
        if st.button("🔄 Generate Time-Lapse Animation", key="timelapse_btn"):
            with st.spinner("Preparing animation..."):
                time_series_data = prepare_time_series_data(
                    raw_data, selected_ticker, pca_model, scaler
                )
                
                if not time_series_data.empty:
                    fig_animation = create_timelapse_animation(
                        time_series_data, selected_ticker, pca_df
                    )
                    st.plotly_chart(fig_animation, use_container_width=True)
                else:
                    st.warning("Insufficient time-series data for animation.")
    
    with tab5:
        st.markdown("### 3D PCA Visualization")
        st.markdown("""
        Explore the clusters in 3D space using the first three principal components.
        Drag to rotate, scroll to zoom.
        """)
        
        if 'PC3' in pca_df.columns:
            fig_3d = create_3d_pca_plot(pca_df, selected_ticker)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D visualization requires PC3 data.")


def render_chatbot_section(
    ticker: str,
    permno: str,
    cluster: int,
    quadrant: str,
    pc1: float,
    pc2: float,
    pca_row: pd.Series,
    percentiles: dict,
    peer_count: int,
    cluster_summary: pd.DataFrame
):
    """Render the AI chatbot section."""
    
    st.markdown("---")
    st.markdown("## 🤖 AI Analysis Assistant")
    
    chatbot = st.session_state.chatbot
    
    if chatbot is None or not chatbot.is_available():
        st.warning("""
        ⚠️ **Chatbot not configured.** 
        
        To enable the AI assistant:
        1. Enter your OpenAI API key in the sidebar
        2. The chatbot will be activated automatically
        
        You can get an API key from [OpenAI's website](https://platform.openai.com/api-keys).
        """)
        return
    
    # Update chatbot context
    factor_data = get_factor_breakdown(pca_row)
    chatbot.set_stock_context(
        ticker=ticker,
        permno=permno,
        cluster=cluster,
        quadrant=quadrant,
        pc1=pc1,
        pc2=pc2,
        factor_data=factor_data,
        percentiles=percentiles,
        peer_count=peer_count,
        cluster_summary=cluster_summary
    )
    
    # Quick analysis button
    if st.button("📝 Get Quick Analysis", key="quick_analysis_btn"):
        analysis = chatbot.get_quick_analysis()
        st.markdown(analysis)
    
    st.markdown("---")
    
    # Sample questions
    st.markdown("### Sample Questions")
    cols = st.columns(4)
    for i, question in enumerate(SAMPLE_QUESTIONS[:4]):
        with cols[i]:
            if st.button(question[:30] + "...", key=f"sample_q_{i}"):
                response = chatbot.get_response(question)
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
    
    # Chat interface
    st.markdown("### Ask a Question")
    
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., How does this stock compare to its peers?",
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send", key="send_btn", type="primary")
    with col2:
        clear_button = st.button("Clear History", key="clear_btn")
    
    if send_button and user_input:
        with st.spinner("Thinking..."):
            response = chatbot.get_response(user_input)
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
    
    if clear_button:
        st.session_state.chat_history = []
        chatbot.clear_history()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main header
    render_main_header()
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data from GitHub..."):
            result = load_and_process_data()
            raw_data, processed_data, pca_df, pca_model, kmeans_model, scaler, error = result
            
            if error:
                st.error(f"❌ Failed to load data: {error}")
                st.markdown(f"""
                **Troubleshooting:**
                1. Check your internet connection
                2. Verify the GitHub URL is accessible: `{GITHUB_DATA_URL}`
                3. Try refreshing the page
                """)
                return
            
            st.session_state.raw_data = raw_data
            st.session_state.processed_data = processed_data
            st.session_state.pca_df = pca_df
            st.session_state.pca_model = pca_model
            st.session_state.kmeans_model = kmeans_model
            st.session_state.scaler = scaler
            st.session_state.data_loaded = True
    
    # Check for selected stock
    if st.session_state.selected_stock is None:
        st.info("👆 Enter a stock ticker or PERMNO in the sidebar to begin analysis.")
        
        # Show overall cluster summary
        if st.session_state.pca_df is not None:
            st.markdown("### 📊 Cluster Overview")
            
            fig = create_pca_scatter_plot(st.session_state.pca_df)
            st.plotly_chart(fig, use_container_width=True)
            
            cluster_summary = get_cluster_summary(st.session_state.pca_df)
            fig_summary = create_cluster_summary_plot(cluster_summary)
            st.plotly_chart(fig_summary, use_container_width=True)
        
        return
    
    # Get selected stock data
    stock_info = st.session_state.selected_stock
    pca_df = st.session_state.pca_df
    
    # Find stock in PCA DataFrame
    if stock_info['type'] == 'ticker':
        mask = pca_df['ticker'].str.upper() == stock_info['value'].upper()
    else:
        mask = pca_df['permno'] == stock_info['value']
    
    stock_pca_data = pca_df[mask]
    
    if stock_pca_data.empty:
        st.error(f"Could not find {stock_info['value']} in the PCA results.")
        return
    
    pca_row = stock_pca_data.iloc[0]
    
    # Render stock overview
    ticker, permno, cluster, pc1, pc2, quadrant = render_stock_overview(
        st.session_state.raw_data, 
        pca_row
    )
    
    # Get quadrant peers
    quadrant_peers = get_stocks_in_same_quadrant(
        pca_df, pc1, pc2, exclude_ticker=ticker
    )
    
    # Render visualizations
    render_visualizations(
        pca_df,
        ticker,
        pca_row,
        quadrant_peers,
        st.session_state.processed_data,
        st.session_state.pca_model,
        st.session_state.scaler
    )
    
    # Get cluster summary and percentiles for chatbot
    cluster_summary = get_cluster_summary(pca_df)
    available_features = [c for c in FEATURE_COLUMNS if c in pca_row.index]
    percentiles = compute_percentile_ranks(quadrant_peers, pca_row, available_features)
    
    # Render chatbot section
    render_chatbot_section(
        ticker, permno, cluster, quadrant, pc1, pc2,
        pca_row, percentiles, len(quadrant_peers), cluster_summary
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
        Stock PCA Cluster Analysis | Built with Streamlit | 
        Data Source: GitHub Repository
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
