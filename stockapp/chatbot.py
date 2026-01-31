"""
OpenAI-powered Chatbot for Stock PCA Cluster Analysis.
Provides context-aware responses about cluster analysis and stock comparisons.
"""

import os
from typing import Dict, List, Optional
import pandas as pd

# Try to import openai - will be needed for the chatbot
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Chatbot will be disabled.")

from config import (
    OPENAI_API_KEY_PLACEHOLDER,
    OPENAI_MODEL,
    CHATBOT_SYSTEM_PROMPT,
    PC1_INTERPRETATION,
    PC2_INTERPRETATION,
    QUADRANTS
)


class StockAnalysisChatbot:
    """
    Chatbot for answering questions about PCA cluster analysis.
    
    Attributes:
        client: OpenAI client instance
        conversation_history: List of conversation messages
        stock_context: Current stock context for responses
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the chatbot with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, tries to get from environment.
        """
        self.client = None
        self.conversation_history: List[Dict] = []
        self.stock_context: Dict = {}
        
        if not OPENAI_AVAILABLE:
            return
        
        # Get API key from parameter, environment, or placeholder
        self.api_key = api_key or os.getenv('OPENAI_API_KEY') or OPENAI_API_KEY_PLACEHOLDER
        
        # Initialize client if we have a valid API key
        if self.api_key and self.api_key != OPENAI_API_KEY_PLACEHOLDER:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
    
    def is_available(self) -> bool:
        """Check if the chatbot is available (OpenAI client initialized)."""
        return self.client is not None
    
    def set_stock_context(
        self,
        ticker: str,
        permno: str,
        cluster: int,
        quadrant: str,
        pc1: float,
        pc2: float,
        factor_data: Dict,
        percentiles: Dict,
        peer_count: int,
        cluster_summary: pd.DataFrame
    ):
        """
        Set the current stock context for the chatbot.
        
        Args:
            ticker: Stock ticker symbol
            permno: PERMNO identifier
            cluster: Cluster number
            quadrant: Quadrant (Q1-Q4)
            pc1: PC1 score
            pc2: PC2 score
            factor_data: Dictionary of factor values
            percentiles: Dictionary of percentile rankings
            peer_count: Number of peers in same quadrant
            cluster_summary: Summary statistics for all clusters
        """
        self.stock_context = {
            'ticker': ticker,
            'permno': permno,
            'cluster': cluster,
            'quadrant': quadrant,
            'pc1': pc1,
            'pc2': pc2,
            'factors': factor_data,
            'percentiles': percentiles,
            'peer_count': peer_count,
            'cluster_summary': cluster_summary.to_dict() if isinstance(cluster_summary, pd.DataFrame) else cluster_summary
        }
        
        # Reset conversation when stock changes
        self.conversation_history = []
    
    def _build_context_message(self) -> str:
        """Build a context message from the current stock data."""
        if not self.stock_context:
            return "No stock is currently selected."
        
        ctx = self.stock_context
        quadrant_info = QUADRANTS.get(ctx['quadrant'], {})
        
        context = f"""
CURRENT STOCK ANALYSIS CONTEXT:

Stock: {ctx['ticker']} (PERMNO: {ctx['permno']})
Cluster: {ctx['cluster']}
Quadrant: {ctx['quadrant']} - {quadrant_info.get('name', 'Unknown')}

PCA Scores:
- PC1 (Quality/Stability): {ctx['pc1']:.3f}
  {'Positive = High quality, profitable, financially strong' if ctx['pc1'] >= 0 else 'Negative = Riskier, lower quality, more volatile'}
- PC2 (Size/Leverage): {ctx['pc2']:.3f}
  {'Positive = Large/liquid, more leveraged' if ctx['pc2'] >= 0 else 'Negative = Cash-rich, operationally efficient'}

Quadrant Characteristics:
{', '.join(quadrant_info.get('characteristics', []))}

Number of peers in same quadrant: {ctx['peer_count']}

Factor Values:
"""
        # Add factor values
        if ctx.get('factors'):
            for category, features in ctx['factors'].items():
                context += f"\n{category}:\n"
                for feature, value in features.items():
                    percentile = ctx.get('percentiles', {}).get(feature, 'N/A')
                    if isinstance(percentile, (int, float)):
                        context += f"  - {feature}: {value:.4f} (Percentile: {percentile:.1f}%)\n"
                    else:
                        context += f"  - {feature}: {value:.4f}\n"
        
        # Add cluster summary
        if ctx.get('cluster_summary'):
            context += "\nCluster Summary (means):\n"
            for cluster_id, data in ctx['cluster_summary'].items():
                if isinstance(data, dict):
                    context += f"Cluster {cluster_id}: "
                    if 'count' in data:
                        context += f"{data.get('count', 'N/A')} stocks\n"
        
        return context
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt with context."""
        base_prompt = CHATBOT_SYSTEM_PROMPT
        
        # Add PC interpretation details
        pc1_details = f"""
PC1 Interpretation ({PC1_INTERPRETATION['name']}):
- Explains ~{PC1_INTERPRETATION['variance_explained']}% of variance
- High values indicate: {', '.join(PC1_INTERPRETATION['high_meaning'])}
- Low values indicate: {', '.join(PC1_INTERPRETATION['low_meaning'])}
"""
        
        pc2_details = f"""
PC2 Interpretation ({PC2_INTERPRETATION['name']}):
- Explains ~{PC2_INTERPRETATION['variance_explained']}% of variance
- High values indicate: {', '.join(PC2_INTERPRETATION['high_meaning'])}
- Low values indicate: {', '.join(PC2_INTERPRETATION['low_meaning'])}
"""
        
        return f"{base_prompt}\n\n{pc1_details}\n{pc2_details}"
    
    def get_response(self, user_message: str) -> str:
        """
        Get a response from the chatbot.
        
        Args:
            user_message: User's question or message
            
        Returns:
            Chatbot's response string
        """
        if not self.is_available():
            return "⚠️ Chatbot is not available. Please configure your OpenAI API key."
        
        # Build messages list
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "system", "content": self._build_context_message()}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return assistant_message
            
        except Exception as e:
            return f"⚠️ Error getting response: {str(e)}"
    
    def get_quick_analysis(self) -> str:
        """
        Get a quick automatic analysis of the current stock.
        
        Returns:
            Quick analysis string
        """
        if not self.stock_context:
            return "No stock selected for analysis."
        
        ctx = self.stock_context
        quadrant_info = QUADRANTS.get(ctx['quadrant'], {})
        
        # Build quick analysis without API call
        analysis = f"""
### Quick Analysis for {ctx['ticker']}

**Cluster Assignment:** Cluster {ctx['cluster']}

**Position:** Quadrant {ctx['quadrant']} - {quadrant_info.get('name', 'Unknown')}

**PCA Interpretation:**
- **PC1 Score ({ctx['pc1']:.3f}):** {'Above average quality/stability' if ctx['pc1'] >= 0 else 'Below average quality/stability'}
- **PC2 Score ({ctx['pc2']:.3f}):** {'Larger/more leveraged' if ctx['pc2'] >= 0 else 'Smaller/more cash-rich'}

**Quadrant Characteristics:**
{chr(10).join(['• ' + c for c in quadrant_info.get('characteristics', [])])}

**Peer Group:** {ctx['peer_count']} other stocks share this quadrant
"""
        return analysis
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []


def create_chatbot(api_key: Optional[str] = None) -> StockAnalysisChatbot:
    """
    Factory function to create a chatbot instance.
    
    Args:
        api_key: Optional OpenAI API key
        
    Returns:
        Configured StockAnalysisChatbot instance
    """
    return StockAnalysisChatbot(api_key=api_key)


# =============================================================================
# SAMPLE QUESTIONS FOR UI
# =============================================================================

SAMPLE_QUESTIONS = [
    "Which cluster does this stock belong to?",
    "How does this stock compare to others in its cluster?",
    "What does the PC1 score tell me about this stock?",
    "Is this stock considered high quality or risky?",
    "What are the key financial characteristics of this stock's cluster?",
    "How does this stock's leverage compare to peers?",
    "What makes this stock different from others in its quadrant?",
    "Should I be concerned about this stock's volatility?",
]
