"""
Stock PCA Analysis Utilities
============================
Utility modules for data loading, visualization, and chatbot functionality.
"""

from .data_loader import (
    load_stock_data,
    load_pca_data,
    load_time_series_pca,
    validate_ticker
)

from .visualizations import (
    create_pca_scatter,
    create_quadrant_analysis,
    create_time_lapse_animation,
    create_factor_breakdown_chart,
    get_quadrant_description
)

from .chatbot import (
    StockAnalysisChatbot,
    format_chat_message,
    get_example_questions
)

__all__ = [
    # Data loading
    'load_stock_data',
    'load_pca_data',
    'load_time_series_pca',
    'validate_ticker',
    
    # Visualizations
    'create_pca_scatter',
    'create_quadrant_analysis',
    'create_time_lapse_animation',
    'create_factor_breakdown_chart',
    'get_quadrant_description',
    
    # Chatbot
    'StockAnalysisChatbot',
    'format_chat_message',
    'get_example_questions'
]
