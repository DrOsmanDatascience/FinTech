"""
Stock PCA Cluster Analysis - Streamlit Application
===================================================
A visualization tool for analyzing stocks in PCA space based on quality,
financial strength, leverage, and liquidity factors.

Author: [Your Name]
Date: January 2026
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_loader import load_stock_data, load_pca_data, validate_ticker
from utils.visualizations import (
    create_pca_scatter,
    create_quadrant_analysis,
    create_time_lapse_animation,
    create_factor_breakdown_chart
)
from utils.chatbot import StockAnalysisChatbot
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Stock PCA Cluster Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main container styling */
    .main {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h4 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Quadrant labels */
    .quadrant-label {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Stock ticker display */
    .ticker-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: white;
        border: 1px solid #e0e0e0;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_pca_plot' not in st.session_state:
    st.session_state.show_pca_plot = False
if 'show_quadrant' not in st.session_state:
    st.session_state.show_quadrant = False
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = False
if 'show_factors' not in st.session_state:
    st.session_state.show_factors = False

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data
def get_data():
    """Load and cache stock data and PCA results."""
    stock_data = load_stock_data()
    pca_data = load_pca_data()
    return stock_data, pca_data

try:
    stock_data, pca_data = get_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False
    stock_data, pca_data = None, None

# =============================================================================
# SIDEBAR - INPUT CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 Stock Selection")
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Ticker Symbol", "PERMNO"],
        horizontal=True
    )
    
    if input_method == "Ticker Symbol":
        user_input = st.text_input(
            "Enter Stock Ticker:",
            placeholder="e.g., AAPL, MSFT, CL",
            help="Enter a valid stock ticker symbol"
        ).upper().strip()
    else:
        user_input = st.text_input(
            "Enter PERMNO:",
            placeholder="e.g., 14593",
            help="Enter a valid PERMNO identifier"
        ).strip()
    
    # Validation and submission
    if st.button("🔍 Analyze Stock", use_container_width=True):
        if user_input:
            if data_loaded:
                is_valid, result = validate_ticker(
                    user_input, 
                    stock_data, 
                    input_type="ticker" if input_method == "Ticker Symbol" else "permno"
                )
                if is_valid:
                    st.session_state.selected_stock = result
                    st.success(f"✅ Found: {result['ticker']} - {result['company_name']}")
                else:
                    st.error(f"❌ {result}")
                    st.session_state.selected_stock = None
            else:
                st.error("Data not loaded. Please check data files.")
        else:
            st.warning("Please enter a ticker or PERMNO")
    
    st.markdown("---")
    st.markdown("### 📈 Visualization Controls")
    
    # Visualization toggle buttons
    if st.button("🎯 Show PCA Scatter Plot", use_container_width=True):
        st.session_state.show_pca_plot = True
    
    if st.button("📍 Show Quadrant Analysis", use_container_width=True):
        st.session_state.show_quadrant = True
    
    if st.button("⏱️ Show Time-Lapse Animation", use_container_width=True):
        st.session_state.show_animation = True
    
    if st.button("📊 Show Factor Breakdown", use_container_width=True):
        st.session_state.show_factors = True
    
    if st.button("🔄 Reset All Views", use_container_width=True):
        st.session_state.show_pca_plot = False
        st.session_state.show_quadrant = False
        st.session_state.show_animation = False
        st.session_state.show_factors = False
    
    st.markdown("---")
    st.markdown("### ℹ️ PCA Interpretation")
    
    with st.expander("PC1: Quality/Stability Axis"):
        st.markdown("""
        **X-Axis (PC1)** explains ~37.5% of variance
        
        **High PC1 (Right):**
        - High-quality
        - Profitable
        - Financially strong
        
        **Low PC1 (Left):**
        - Riskier
        - Lower-quality
        - Volatile
        """)
    
    with st.expander("PC2: Size/Leverage Axis"):
        st.markdown("""
        **Y-Axis (PC2)** explains ~14.6% of variance
        
        **High PC2 (Top):**
        - Large / Liquid
        - Leveraged
        
        **Low PC2 (Bottom):**
        - Cash-rich
        - Operationally efficient
        """)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================
st.markdown("# 📊 Stock PCA Cluster Analysis")
st.markdown("*Visualize stocks across quality, financial strength, and capital structure dimensions*")

# Display selected stock information
if st.session_state.selected_stock:
    stock = st.session_state.selected_stock
    
    st.markdown("---")
    
    # Stock header with key metrics
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span class="ticker-display">{stock['ticker']}</span>
            <br>
            <span style="font-size: 1.2rem; color: #666;">{stock['company_name']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="PC1 Score",
            value=f"{stock['pc1']:.2f}",
            delta="Quality/Stability"
        )
    
    with col3:
        st.metric(
            label="PC2 Score",
            value=f"{stock['pc2']:.2f}",
            delta="Size/Leverage"
        )
    
    with col4:
        st.metric(
            label="Cluster",
            value=f"Cluster {stock['cluster']}",
            delta=stock['quadrant_name']
        )
    
    st.markdown("---")

# =============================================================================
# VISUALIZATION TABS
# =============================================================================
if data_loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 PCA Scatter Plot",
        "📍 Quadrant Analysis", 
        "⏱️ Time-Lapse Animation",
        "📊 Factor Breakdown",
        "🤖 AI Assistant"
    ])
    
    # ----- TAB 1: PCA SCATTER PLOT -----
    with tab1:
        st.markdown("### Principal Component Analysis - Stock Positioning")
        st.markdown("""
        This visualization shows all stocks plotted in PCA space. The X-axis represents 
        quality/stability factors while the Y-axis represents size/leverage characteristics.
        """)
        
        if st.session_state.show_pca_plot or st.button("Generate PCA Plot", key="pca_btn"):
            st.session_state.show_pca_plot = True
            
            # Create PCA scatter plot
            fig = create_pca_scatter(
                pca_data, 
                selected_stock=st.session_state.selected_stock
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quadrant legend
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="quadrant-label">
                    <strong>Top-Right:</strong> High-quality, Large/Liquid<br>
                    <small>Profitable, financially strong, leveraged companies</small>
                </div>
                <div class="quadrant-label">
                    <strong>Bottom-Right:</strong> High-quality, Cash-rich<br>
                    <small>Profitable, operationally efficient companies</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="quadrant-label">
                    <strong>Top-Left:</strong> Lower-quality, Large/Liquid<br>
                    <small>Riskier, volatile, leveraged companies</small>
                </div>
                <div class="quadrant-label">
                    <strong>Bottom-Left:</strong> Lower-quality, Cash-rich<br>
                    <small>Riskier, volatile, smaller companies</small>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- TAB 2: QUADRANT ANALYSIS -----
    with tab2:
        st.markdown("### Quadrant Peer Comparison")
        
        if st.session_state.selected_stock:
            if st.session_state.show_quadrant or st.button("Show Quadrant Peers", key="quad_btn"):
                st.session_state.show_quadrant = True
                
                stock = st.session_state.selected_stock
                
                # Get quadrant analysis
                fig, peer_df = create_quadrant_analysis(
                    pca_data, 
                    stock
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show peer table
                st.markdown("#### Stocks in Same Quadrant")
                st.markdown(f"*{stock['ticker']} is in the **{stock['quadrant_name']}** quadrant*")
                
                if not peer_df.empty:
                    # Calculate market weight rank
                    peer_df_display = peer_df[['ticker', 'company_name', 'market_weight', 'pc1', 'pc2']].copy()
                    peer_df_display['Market Weight Rank'] = peer_df_display['market_weight'].rank(ascending=False).astype(int)
                    peer_df_display = peer_df_display.sort_values('market_weight', ascending=False)
                    
                    st.dataframe(
                        peer_df_display.rename(columns={
                            'ticker': 'Ticker',
                            'company_name': 'Company',
                            'market_weight': 'Market Weight (%)',
                            'pc1': 'PC1 Score',
                            'pc2': 'PC2 Score'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Show selected stock's rank
                    stock_rank = peer_df_display[peer_df_display['ticker'] == stock['ticker']]['Market Weight Rank'].values
                    if len(stock_rank) > 0:
                        st.info(f"📊 {stock['ticker']} ranks **#{int(stock_rank[0])}** out of **{len(peer_df_display)}** stocks in its quadrant by market weight")
        else:
            st.info("👆 Please select a stock from the sidebar to see quadrant analysis")
    
    # ----- TAB 3: TIME-LAPSE ANIMATION -----
    with tab3:
        st.markdown("### Historical Position Animation")
        st.markdown("""
        Watch how the selected stock's position has evolved over time in PCA space.
        This reveals trends in quality, financial strength, and capital structure.
        """)
        
        if st.session_state.selected_stock:
            if st.session_state.show_animation or st.button("▶️ Run Time-Lapse Animation", key="anim_btn"):
                st.session_state.show_animation = True
                
                stock = st.session_state.selected_stock
                
                # Create animation
                fig = create_time_lapse_animation(
                    pca_data,
                    stock['ticker']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                    <strong>How to interpret:</strong>
                    <ul>
                        <li>Use the play button to animate the stock's movement over time</li>
                        <li>The trail shows historical positions</li>
                        <li>Movement right → improving quality/stability</li>
                        <li>Movement up → increasing size/leverage</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Please select a stock from the sidebar to see the time-lapse animation")
    
    # ----- TAB 4: FACTOR BREAKDOWN -----
    with tab4:
        st.markdown("### Factor Component Analysis")
        st.markdown("""
        Detailed breakdown of the underlying factors that determine the stock's 
        position in PCA space.
        """)
        
        if st.session_state.selected_stock:
            if st.session_state.show_factors or st.button("Show Factor Analysis", key="factor_btn"):
                st.session_state.show_factors = True
                
                stock = st.session_state.selected_stock
                
                # Create factor breakdown charts
                fig_radar, fig_bar = create_factor_breakdown_chart(
                    stock,
                    pca_data
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Factor Radar Chart")
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    st.markdown("#### Percentile Rankings")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Factor details table
                st.markdown("#### Detailed Factor Scores")
                
                factor_data = {
                    'Factor': ['Value', 'Quality', 'Financial Strength', 'Momentum', 'Volatility', 'Liquidity'],
                    'Score': [
                        stock.get('value_score', 0),
                        stock.get('quality_score', 0),
                        stock.get('fin_strength_score', 0),
                        stock.get('momentum_score', 0),
                        stock.get('volatility_score', 0),
                        stock.get('liquidity_score', 0)
                    ],
                    'Percentile (vs Quadrant)': [
                        stock.get('value_pctl', 50),
                        stock.get('quality_pctl', 50),
                        stock.get('fin_strength_pctl', 50),
                        stock.get('momentum_pctl', 50),
                        stock.get('volatility_pctl', 50),
                        stock.get('liquidity_pctl', 50)
                    ]
                }
                
                st.dataframe(
                    pd.DataFrame(factor_data),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("👆 Please select a stock from the sidebar to see factor breakdown")
    
    # ----- TAB 5: AI CHATBOT -----
    with tab5:
        st.markdown("### 🤖 AI Stock Analysis Assistant")
        st.markdown("""
        Ask questions about the cluster analysis, selected stock, and visualizations.
        The assistant has context about the current analysis.
        """)
        
        # Initialize chatbot
        chatbot = StockAnalysisChatbot()
        
        # Check for API key (Streamlit Cloud uses st.secrets, local uses env vars)
        api_key = ""
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not api_key:
            st.warning("""
            ⚠️ **OpenAI API Key Required**
            
            To use the AI assistant, please set your OpenAI API key:
            1. Create a `.env` file in the project root
            2. Add: `OPENAI_API_KEY=your-api-key-here`
            3. Or set it as an environment variable
            """)
            
            # Manual key input for testing
            api_key = st.text_input("Or enter API key here:", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        
        # Build context for chatbot
        context = chatbot.build_context(
            selected_stock=st.session_state.selected_stock,
            pca_data=pca_data
        )
        
        # Display context summary
        with st.expander("📋 Current Analysis Context"):
            if st.session_state.selected_stock:
                stock = st.session_state.selected_stock
                st.markdown(f"""
                **Selected Stock:** {stock['ticker']} - {stock['company_name']}
                - PC1 Score: {stock['pc1']:.2f} (Quality/Stability)
                - PC2 Score: {stock['pc2']:.2f} (Size/Leverage)
                - Cluster: {stock['cluster']}
                - Quadrant: {stock['quadrant_name']}
                """)
            else:
                st.markdown("*No stock selected - general questions only*")
        
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., Which cluster does this stock belong to?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", use_container_width=True)
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Get response from chatbot
            with st.spinner("Thinking..."):
                response = chatbot.get_response(
                    user_question,
                    context,
                    st.session_state.chat_history
                )
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
        
        # Example questions
        st.markdown("---")
        st.markdown("**Example questions you can ask:**")
        example_qs = [
            "Which cluster does this stock belong to?",
            "How does this stock compare to others in its cluster?",
            "What factors contribute most to this stock's position?",
            "What are the characteristics of stocks in the top-right quadrant?",
            "Is this stock considered high-quality based on the PCA analysis?"
        ]
        
        for q in example_qs:
            if st.button(q, key=f"example_{q[:20]}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                response = chatbot.get_response(q, context, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

else:
    st.error("""
    ⚠️ **Data Not Loaded**
    
    Please ensure the following files exist in the `data/` directory:
    - `stock_data.csv` - Stock information and factors
    - `pca_results.csv` - PCA scores and cluster assignments
    - `time_series_pca.csv` - Historical PCA positions
    
    See the README for data format requirements.
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Stock PCA Cluster Analysis Tool | Built with Streamlit</p>
    <p>PC1 + PC2 explain 52.5% of total variance</p>
</div>
""", unsafe_allow_html=True)
