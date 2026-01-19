"""
Stock Analysis Chatbot
======================
OpenAI-powered chatbot for answering questions about PCA cluster analysis.
"""

import os
from typing import Dict, List, Optional
import json

# =============================================================================
# CHATBOT CLASS
# =============================================================================

class StockAnalysisChatbot:
    """
    AI-powered chatbot for stock PCA analysis questions.
    
    The chatbot is restricted to answering questions about:
    - Cluster analysis results
    - Currently selected stock
    - Displayed visualizations
    - Factor interpretations
    """
    
    def __init__(self):
        """Initialize the chatbot."""
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = "gpt-4o-mini"  # Can change to "gpt-4" for better responses
        
        # System prompt defining chatbot behavior and knowledge
        self.system_prompt = """You are an expert financial analyst assistant specialized in PCA (Principal Component Analysis) cluster analysis for stocks. You help users understand stock positioning, cluster characteristics, and factor analysis.

## Your Knowledge Base:

### PCA Framework:
- **PC1 (X-axis)** explains ~37.5% of variance and represents "Quality / Stability / Balance-Sheet Strength"
  - Strong positive loadings: ROA (0.44), ROE (0.36), Earnings Yield (0.39), Cash/Debt (0.36), Momentum (0.26)
  - Strong negative loadings: Volatility (-0.34), Book-to-Market (-0.35)
  - High PC1 = High-quality, profitable, financially strong
  - Low PC1 = Riskier, lower-quality, volatile

- **PC2 (Y-axis)** explains ~14.6% of variance and represents "Size / Leverage / Capital Structure"
  - Strong positive loadings: ADDV/liquidity (0.54), Debt-to-Assets (0.48)
  - Strong negative loadings: Cash/Debt (-0.34), Sales-to-Price (-0.29), Gross Profitability (-0.27)
  - High PC2 = Large/liquid, leveraged
  - Low PC2 = Cash-rich, operationally efficient

- Together PC1 & PC2 explain 52.5% of total variance

### Quadrants:
1. **Top-Right (Cluster 0)**: High-quality + Large/Liquid
   - Profitable, financially strong companies
   - Higher leverage but manageable due to quality
   - Examples: Blue-chip consumer staples, large-cap tech leaders

2. **Top-Left (Cluster 1)**: Lower-quality + Large/Liquid  
   - Growth stocks, more volatile
   - High trading volume, leveraged
   - Examples: High-growth tech, momentum stocks

3. **Bottom-Right (Cluster 2)**: High-quality + Cash-rich
   - Profitable with strong balance sheets
   - Lower leverage, operationally efficient
   - Examples: Cash-rich tech giants, defensive healthcare

4. **Bottom-Left (Cluster 3)**: Lower-quality + Cash-rich
   - Riskier, more speculative
   - Smaller or earlier-stage companies
   - Examples: Speculative growth, turnaround plays

### Key Factors:
- **Value**: Earnings Yield, Book-to-Market, Sales-to-Price
- **Quality**: ROA, ROE, Gross Profitability
- **Financial Strength**: Cash/Debt ratio, Debt-to-Assets
- **Momentum**: Recent price performance
- **Volatility**: Price variability (inverse relationship with quality)
- **Liquidity**: Average daily dollar volume (ADDV)

## Response Guidelines:
1. Always relate answers back to the PCA framework
2. Use specific factor loadings when explaining positions
3. Provide actionable insights when discussing stocks
4. Be clear about what the analysis does and doesn't tell us
5. Acknowledge limitations (PCA captures 52.5% of variance, not everything)
6. Use simple language while maintaining accuracy
7. Keep responses concise but informative

## Important Limitations:
- You cannot provide investment advice or recommendations
- The analysis is based on historical data and factor exposures
- Cluster membership can change over time as fundamentals change
- Always recommend users do additional research"""
    
    def build_context(self, selected_stock: Optional[Dict], pca_data) -> str:
        """
        Build context string from current analysis state.
        
        Parameters:
        -----------
        selected_stock : dict or None
            Currently selected stock information
        pca_data : pd.DataFrame
            Full PCA dataset
        
        Returns:
        --------
        str
            Context string for the chatbot
        """
        context_parts = []
        
        # Add selected stock context
        if selected_stock:
            context_parts.append(f"""
## Currently Selected Stock:
- **Ticker**: {selected_stock['ticker']}
- **Company**: {selected_stock.get('company_name', 'N/A')}
- **Sector**: {selected_stock.get('sector', 'N/A')}
- **PC1 Score**: {selected_stock.get('pc1', 0):.2f} (Quality/Stability axis)
- **PC2 Score**: {selected_stock.get('pc2', 0):.2f} (Size/Leverage axis)
- **Cluster**: {selected_stock.get('cluster', 'N/A')} - {selected_stock.get('quadrant_name', 'N/A')}
- **Market Weight**: {selected_stock.get('market_weight', 0):.2f}%

### Factor Scores:
- Value: {selected_stock.get('value_score', 0.5):.2f}
- Quality: {selected_stock.get('quality_score', 0.5):.2f}
- Financial Strength: {selected_stock.get('fin_strength_score', 0.5):.2f}
- Momentum: {selected_stock.get('momentum_score', 0.5):.2f}
- Volatility: {selected_stock.get('volatility_score', 0.5):.2f}
- Liquidity: {selected_stock.get('liquidity_score', 0.5):.2f}
""")
            
            # Add cluster peer context
            if pca_data is not None:
                cluster = selected_stock.get('cluster')
                if cluster is not None:
                    cluster_peers = pca_data[pca_data['cluster'] == cluster]
                    peer_count = len(cluster_peers)
                    peer_tickers = cluster_peers['ticker'].tolist()[:10]  # Limit to 10
                    
                    context_parts.append(f"""
### Cluster Peers ({peer_count} total):
Top peers in same quadrant: {', '.join(peer_tickers)}
""")
        else:
            context_parts.append("""
## No Stock Currently Selected
The user has not yet selected a specific stock. They may ask general questions about the PCA framework or cluster characteristics.
""")
        
        # Add summary statistics if data available
        if pca_data is not None:
            cluster_counts = pca_data['cluster'].value_counts().to_dict()
            context_parts.append(f"""
### Dataset Summary:
- Total stocks in analysis: {len(pca_data)}
- Cluster 0 (High-Quality/Large): {cluster_counts.get(0, 0)} stocks
- Cluster 1 (Lower-Quality/Large): {cluster_counts.get(1, 0)} stocks
- Cluster 2 (High-Quality/Cash-Rich): {cluster_counts.get(2, 0)} stocks
- Cluster 3 (Lower-Quality/Smaller): {cluster_counts.get(3, 0)} stocks
""")
        
        return "\n".join(context_parts)
    
    def get_response(self, user_question: str, context: str, 
                     chat_history: List[Dict]) -> str:
        """
        Get chatbot response using OpenAI API.
        
        Parameters:
        -----------
        user_question : str
            User's question
        context : str
            Current analysis context
        chat_history : list
            Previous messages in conversation
        
        Returns:
        --------
        str
            Chatbot response
        """
        # Check for API key (Streamlit Cloud uses st.secrets, local uses env vars)
        api_key = ""
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            pass
        
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not api_key:
            return self._get_fallback_response(user_question, context)
        
        try:
            # Import OpenAI (only when needed)
            from openai import OpenAI
            
            client = OpenAI(api_key=api_key)
            
            # Build messages array
            messages = [
                {"role": "system", "content": self.system_prompt + "\n\n" + context}
            ]
            
            # Add recent chat history (last 6 messages max)
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current question
            messages.append({"role": "user", "content": user_question})
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return "⚠️ OpenAI library not installed. Please run: `pip install openai`"
        except Exception as e:
            return f"⚠️ Error getting response: {str(e)}\n\nUsing fallback response:\n\n{self._get_fallback_response(user_question, context)}"
    
    def _get_fallback_response(self, question: str, context: str) -> str:
        """
        Provide rule-based fallback responses when API is unavailable.
        
        Parameters:
        -----------
        question : str
            User's question
        context : str
            Current analysis context
        
        Returns:
        --------
        str
            Fallback response
        """
        question_lower = question.lower()
        
        # Parse context for stock info
        stock_ticker = None
        stock_cluster = None
        if "Ticker**: " in context:
            try:
                stock_ticker = context.split("Ticker**: ")[1].split("\n")[0].strip()
            except:
                pass
        if "Cluster**: " in context:
            try:
                cluster_info = context.split("Cluster**: ")[1].split("\n")[0]
                stock_cluster = int(cluster_info.split(" -")[0])
            except:
                pass
        
        # Cluster questions
        if "cluster" in question_lower and "belong" in question_lower:
            if stock_ticker and stock_cluster is not None:
                cluster_names = {
                    0: "High Quality / Large-Liquid (top-right quadrant)",
                    1: "Lower Quality / Large-Liquid (top-left quadrant)",
                    2: "High Quality / Cash-Rich (bottom-right quadrant)",
                    3: "Lower Quality / Smaller (bottom-left quadrant)"
                }
                return f"Based on the PCA analysis, **{stock_ticker}** belongs to **Cluster {stock_cluster}**: {cluster_names.get(stock_cluster, 'Unknown')}.\n\nThis means the stock exhibits characteristics consistent with that quadrant's profile based on quality, financial strength, and capital structure factors."
            else:
                return "Please select a stock first to see its cluster assignment."
        
        # Comparison questions
        if "compare" in question_lower or "how does" in question_lower:
            if stock_ticker:
                return f"To compare {stock_ticker} with its cluster peers:\n\n1. Check the **Quadrant Analysis** tab to see all stocks in the same quadrant\n2. Look at the **Factor Breakdown** tab to see percentile rankings vs peers\n3. The radar chart shows how {stock_ticker}'s factor scores compare to the cluster average\n\nKey factors to compare: Value, Quality, Financial Strength, Momentum, Volatility, and Liquidity."
            else:
                return "Please select a stock to compare it with its cluster peers."
        
        # Factor questions
        if "factor" in question_lower or "contribut" in question_lower:
            return """The main factors that determine a stock's position in PCA space are:

**PC1 (Quality/Stability) contributors:**
- ROA and ROE (profitability) - positive loading
- Earnings Yield (value) - positive loading
- Cash/Debt ratio (financial strength) - positive loading
- Momentum - positive loading
- Volatility - negative loading
- Book-to-Market - negative loading

**PC2 (Size/Leverage) contributors:**
- Average Daily Dollar Volume (liquidity) - positive loading
- Debt-to-Assets (leverage) - positive loading
- Cash/Debt - negative loading
- Sales-to-Price - negative loading
- Gross Profitability - negative loading

Check the Factor Breakdown tab for the selected stock's specific scores."""
        
        # Quadrant characteristics
        if "quadrant" in question_lower or "characteristic" in question_lower:
            if "top-right" in question_lower or "high quality" in question_lower and "large" in question_lower:
                return "**Top-Right Quadrant (High-Quality / Large-Liquid):**\n\nCharacteristics:\n- Higher profitability (ROA, ROE)\n- Strong earnings yield\n- Good cash/debt ratios\n- Positive momentum\n- Lower volatility\n- Higher trading volume/liquidity\n- More leveraged capital structure\n\nTypical stocks: Blue-chip consumer staples, large-cap tech leaders with strong fundamentals"
            elif "top-left" in question_lower or "lower quality" in question_lower and "large" in question_lower:
                return "**Top-Left Quadrant (Lower-Quality / Large-Liquid):**\n\nCharacteristics:\n- Lower profitability metrics\n- Higher volatility\n- Growth-oriented characteristics\n- High trading volume/liquidity\n- More leveraged capital structure\n\nTypical stocks: High-growth tech, momentum stocks, companies prioritizing growth over profitability"
            elif "bottom-right" in question_lower or "cash-rich" in question_lower and "high quality" in question_lower:
                return "**Bottom-Right Quadrant (High-Quality / Cash-Rich):**\n\nCharacteristics:\n- Higher profitability (ROA, ROE)\n- Strong balance sheets\n- Lower leverage / more cash\n- Operationally efficient\n- Lower volatility\n\nTypical stocks: Cash-rich tech giants, defensive healthcare companies"
            else:
                return "The PCA space has four quadrants:\n\n1. **Top-Right**: High-quality, profitable, large/liquid companies\n2. **Top-Left**: Growth-oriented, volatile, large/liquid companies\n3. **Bottom-Right**: High-quality, cash-rich, operationally efficient\n4. **Bottom-Left**: Smaller, speculative, cash-rich companies\n\nEach quadrant represents different factor exposures. Ask about a specific quadrant for more details."
        
        # High quality questions
        if "high-quality" in question_lower or "high quality" in question_lower:
            if stock_ticker:
                return f"A stock is considered 'high-quality' based on the PCA analysis when it has a **positive PC1 score**, indicating:\n- Higher profitability (ROA, ROE)\n- Strong earnings yield\n- Good financial strength (cash/debt ratio)\n- Lower volatility\n- Positive momentum\n\nCheck the Factor Breakdown tab to see {stock_ticker}'s specific quality metrics and how they compare to peers."
        
        # Default response
        return """I can help you understand the PCA cluster analysis! Here are some questions I can answer:

- "Which cluster does this stock belong to?"
- "How does this stock compare to others in its cluster?"
- "What factors contribute to this stock's position?"
- "What are the characteristics of the top-right quadrant?"
- "Is this stock considered high-quality?"

Please select a stock from the sidebar to get specific insights about that company."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_chat_message(role: str, content: str) -> Dict:
    """Format a chat message."""
    return {"role": role, "content": content}


def get_example_questions() -> List[str]:
    """Get list of example questions users can ask."""
    return [
        "Which cluster does this stock belong to?",
        "How does this stock compare to others in its cluster?",
        "What factors contribute most to this stock's position?",
        "What are the characteristics of the top-right quadrant?",
        "Is this stock considered high-quality based on the analysis?",
        "Why is this stock positioned where it is on the chart?",
        "What would cause this stock to move to a different quadrant?",
        "How volatile is this stock compared to its peers?",
        "What does a high PC1 score indicate?",
        "Explain the difference between the quadrants."
    ]


if __name__ == "__main__":
    # Test chatbot
    print("Testing chatbot...")
    
    chatbot = StockAnalysisChatbot()
    
    # Test context building
    test_stock = {
        'ticker': 'AAPL',
        'company_name': 'Apple Inc',
        'sector': 'Technology',
        'pc1': 2.5,
        'pc2': 0.8,
        'cluster': 0,
        'quadrant_name': 'High Quality / Large-Liquid',
        'market_weight': 7.1,
        'value_score': 0.65,
        'quality_score': 0.85,
        'fin_strength_score': 0.78,
        'momentum_score': 0.72,
        'volatility_score': 0.35,
        'liquidity_score': 0.95
    }
    
    context = chatbot.build_context(test_stock, None)
    print("Context built successfully")
    print(context[:500] + "...")
    
    # Test fallback responses
    test_questions = [
        "Which cluster does this stock belong to?",
        "What are the characteristics of the top-right quadrant?",
        "What factors contribute to the position?"
    ]
    
    print("\nTesting fallback responses:")
    for q in test_questions:
        print(f"\nQ: {q}")
        response = chatbot._get_fallback_response(q, context)
        print(f"A: {response[:200]}...")
    
    print("\nChatbot tests completed!")
