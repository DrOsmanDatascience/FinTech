"""
Configuration Management
========================
Centralized configuration for the Stock PCA Analysis application.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Toggle between sample data and real data
USE_SAMPLE_DATA = True  # Set to False to use real data files

# GitHub repository for data (update with your repo)
GITHUB_CONFIG = {
    "username": "YOUR_GITHUB_USERNAME",
    "repository": "YOUR_REPO_NAME",
    "branch": "main",
    "data_folder": "data"
}

# Construct GitHub raw URL
GITHUB_BASE_URL = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_CONFIG['username']}/"
    f"{GITHUB_CONFIG['repository']}/"
    f"{GITHUB_CONFIG['branch']}/"
    f"{GITHUB_CONFIG['data_folder']}"
)

# Data file names
DATA_FILES = {
    "stock_data": "stock_data.csv",
    "pca_results": "pca_results.csv",
    "time_series": "time_series_pca.csv"
}

# =============================================================================
# PCA CONFIGURATION
# =============================================================================

# PCA loadings (from the visualization guide)
PC1_LOADINGS = {
    "ROA": 0.44,
    "ROE": 0.36,
    "earnings_yield": 0.39,
    "cash_debt": 0.36,
    "momentum": 0.26,
    "volatility": -0.34,
    "book_to_market": -0.35
}

PC2_LOADINGS = {
    "ADDV": 0.54,
    "debt_to_assets": 0.48,
    "cash_debt": -0.34,
    "sales_to_price": -0.29,
    "gross_profitability": -0.27
}

# Variance explained
PC1_VARIANCE = 0.375  # 37.5%
PC2_VARIANCE = 0.146  # 14.6%
TOTAL_VARIANCE_EXPLAINED = PC1_VARIANCE + PC2_VARIANCE  # 52.5%

# Cluster names
CLUSTER_NAMES = {
    0: "High Quality / Large-Liquid",
    1: "Lower Quality / Large-Liquid", 
    2: "High Quality / Cash-Rich",
    3: "Lower Quality / Smaller"
}

# Quadrant characteristics
QUADRANT_CHARACTERISTICS = {
    0: {
        "pc1": "High (>0)",
        "pc2": "High (>0)",
        "description": "High-quality, profitable, large/liquid companies",
        "traits": ["Profitable", "Financially strong", "Higher leverage", "High liquidity"]
    },
    1: {
        "pc1": "Low (<0)",
        "pc2": "High (>0)",
        "description": "Growth-oriented, volatile, large/liquid companies",
        "traits": ["Growth-focused", "Higher volatility", "Leveraged", "High trading volume"]
    },
    2: {
        "pc1": "High (>0)",
        "pc2": "Low (<0)",
        "description": "High-quality, cash-rich, operationally efficient companies",
        "traits": ["Profitable", "Strong balance sheet", "Low leverage", "Efficient operations"]
    },
    3: {
        "pc1": "Low (<0)",
        "pc2": "Low (<0)",
        "description": "Speculative, smaller, cash-rich companies",
        "traits": ["Higher risk", "Smaller size", "More volatile", "Speculative"]
    }
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Color scheme
COLORS = {
    "cluster_0": "#667eea",  # Purple-blue (High Quality / Large-Liquid)
    "cluster_1": "#f6ad55",  # Orange (Lower Quality / Large-Liquid)
    "cluster_2": "#48bb78",  # Green (High Quality / Cash-Rich)
    "cluster_3": "#fc8181",  # Red (Lower Quality / Smaller)
    "selected": "#1a1a2e",   # Dark (selected stock)
    "grid": "#e2e8f0",       # Light gray (grid lines)
    "text": "#2d3748",       # Dark gray (text)
    "background": "#ffffff"  # White (background)
}

# Chart dimensions
CHART_CONFIG = {
    "pca_scatter": {"height": 600, "width": None},  # None = use container width
    "quadrant_analysis": {"height": 500, "width": None},
    "time_lapse": {"height": 550, "width": None},
    "factor_radar": {"height": 350, "width": None},
    "factor_bar": {"height": 350, "width": None}
}

# Animation settings
ANIMATION_CONFIG = {
    "frame_duration": 300,  # milliseconds per frame
    "transition_duration": 200,
    "periods": 24  # months of history
}

# =============================================================================
# CHATBOT CONFIGURATION
# =============================================================================

# OpenAI settings
OPENAI_CONFIG = {
    "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "max_tokens": 500,
    "temperature": 0.7,
    "max_history_messages": 6  # Keep last N messages for context
}

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

APP_CONFIG = {
    "title": "Stock PCA Cluster Analysis",
    "icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_data_path(file_key: str) -> Path:
    """Get the full path to a data file."""
    if file_key in DATA_FILES:
        return DATA_DIR / DATA_FILES[file_key]
    raise ValueError(f"Unknown data file key: {file_key}")


def get_github_url(file_key: str) -> str:
    """Get the GitHub raw URL for a data file."""
    if file_key in DATA_FILES:
        return f"{GITHUB_BASE_URL}/{DATA_FILES[file_key]}"
    raise ValueError(f"Unknown data file key: {file_key}")


def get_cluster_color(cluster: int) -> str:
    """Get the color for a specific cluster."""
    return COLORS.get(f"cluster_{cluster}", COLORS["cluster_0"])


def get_quadrant_from_scores(pc1: float, pc2: float) -> int:
    """Determine quadrant/cluster from PC scores."""
    if pc1 >= 0 and pc2 >= 0:
        return 0
    elif pc1 < 0 and pc2 >= 0:
        return 1
    elif pc1 >= 0 and pc2 < 0:
        return 2
    else:
        return 3


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    issues = []
    
    # Check if data directory exists (if using real data)
    if not USE_SAMPLE_DATA and not DATA_DIR.exists():
        issues.append(f"Data directory not found: {DATA_DIR}")
    
    # Check OpenAI key if chatbot is expected to work
    if not os.environ.get("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set (chatbot will use fallback responses)")
    
    return issues


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Stock PCA Analysis - Configuration Summary")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Use Sample Data: {USE_SAMPLE_DATA}")
    print(f"\nPCA Variance Explained:")
    print(f"  PC1: {PC1_VARIANCE * 100:.1f}%")
    print(f"  PC2: {PC2_VARIANCE * 100:.1f}%")
    print(f"  Total: {TOTAL_VARIANCE_EXPLAINED * 100:.1f}%")
    print(f"\nClusters:")
    for cluster_id, name in CLUSTER_NAMES.items():
        print(f"  {cluster_id}: {name}")
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print(f"\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration validated successfully")
