# Stock PCA Cluster Analysis

An interactive Streamlit web application for visualizing stocks in PCA (Principal Component Analysis) space, analyzing cluster characteristics, and exploring factor breakdowns.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)

## Quick Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your forked repo → Deploy!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Overview

This application provides a visual framework for understanding stock positioning based on:

- **PC1 (X-axis)**: Quality / Stability / Balance-Sheet Strength (~37.5% of variance)
- **PC2 (Y-axis)**: Size / Leverage / Capital Structure (~14.6% of variance)

Together, these components explain **52.5% of total variance** in stock factor characteristics.

### Quadrant Interpretation

| Quadrant | PC1 | PC2 | Characteristics |
|----------|-----|-----|-----------------|
| Top-Right | High | High | High-quality, profitable, large/liquid |
| Top-Left | Low | High | Growth-oriented, volatile, large/liquid |
| Bottom-Right | High | Low | High-quality, cash-rich, operationally efficient |
| Bottom-Left | Low | Low | Speculative, smaller, cash-rich |

## Features

### 1. Interactive PCA Scatter Plot
- View all stocks plotted in PCA space
- Color-coded by cluster assignment
- Hover for detailed stock information
- Highlight selected stock with diamond marker

### 2. Quadrant Peer Analysis
- Focus on stocks in the same quadrant as selected stock
- Market weight rankings within quadrant
- Peer comparison table

### 3. Time-Lapse Animation
- Watch stock movement over time (24 months)
- Animated path through PCA space
- Play/pause controls and time slider

### 4. Factor Breakdown
- Radar chart comparing to cluster average
- Percentile rankings vs quadrant peers
- Detailed factor score table

### 5. AI-Powered Chatbot
- OpenAI-powered analysis assistant
- Contextual answers about cluster analysis
- Fallback responses when API unavailable

## 📁 Project Structure

```
stock_pca_app/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── __init__.py       # Package initialization
│   ├── data_loader.py    # Data loading and validation
│   ├── visualizations.py # Plotly chart generation
│   └── chatbot.py        # OpenAI chatbot implementation
├── data/                  # Data directory (create if using real data)
│   ├── stock_data.csv    # Stock information and factors
│   ├── pca_results.csv   # PCA scores and cluster assignments
│   └── time_series_pca.csv # Historical PCA positions
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- VS Code (recommended) or any Python IDE

### Setup

1. **Clone/Download the repository**
   ```bash
   # If using GitHub
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd stock_pca_app
   
   # Or create the directory manually
   mkdir stock_pca_app
   cd stock_pca_app
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## Data Configuration

### Using Sample Data (Default)

The application includes built-in sample data generation for demonstration purposes. No additional setup required.

### Using Real Data

To use your own data from a GitHub repository:

1. **Update `utils/data_loader.py`**:
   
   ```python
   # Line ~15: Update the GitHub URL
   GITHUB_BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data"
   
   # Line ~45: Set use_sample=False
   def load_stock_data(use_sample=False):
       ...
   ```

2. **Required Data Format**:

   **stock_data.csv**:
   ```csv
   ticker,permno,company_name,sector,market_weight,value_score,quality_score,fin_strength_score,momentum_score,volatility_score,liquidity_score
   AAPL,14593,Apple Inc,Technology,7.1,0.65,0.85,0.78,0.72,0.35,0.95
   ...
   ```

   **pca_results.csv**:
   ```csv
   ticker,permno,company_name,pc1,pc2,cluster
   AAPL,14593,Apple Inc,2.5,0.8,0
   ...
   ```

   **time_series_pca.csv** (for animations):
   ```csv
   date,ticker,pc1,pc2
   2024-01-31,AAPL,2.3,0.7
   2024-02-29,AAPL,2.4,0.75
   ...
   ```

## Chatbot Configuration

The AI chatbot uses OpenAI's GPT models. Features:

### With API Key
- Full conversational AI capabilities
- Context-aware responses about selected stock
- Multi-turn conversations

### Without API Key (Fallback)
- Rule-based responses for common questions
- Still provides helpful information about PCA framework
- No external API calls required

### Configuring the Model

In `utils/chatbot.py`, you can change the model:

```python
self.model = "gpt-4o-mini"  # Default: faster, cheaper
# self.model = "gpt-4"       # Alternative: more capable
# self.model = "gpt-3.5-turbo"  # Alternative: fastest
```

## Customization

### Colors and Styling

Edit the `COLORS` dictionary in `utils/visualizations.py`:

```python
COLORS = {
    'cluster_0': '#667eea',  # Purple-blue
    'cluster_1': '#f6ad55',  # Orange
    'cluster_2': '#48bb78',  # Green
    'cluster_3': '#fc8181',  # Red
    ...
}
```

### Adding New Visualizations

1. Create a new function in `utils/visualizations.py`
2. Import it in `utils/__init__.py`
3. Add a new tab in `app.py`

## Usage Examples

### Basic Stock Analysis
1. Enter a ticker (e.g., "AAPL") in the sidebar
2. Click "Analyze Stock"
3. Navigate through tabs to see different visualizations

### Peer Comparison
1. Select a stock
2. Go to "Quadrant Analysis" tab
3. View all stocks in the same cluster
4. Check market weight rankings

### AI Assistant
1. Go to "AI Assistant" tab
2. Ask questions like:
   - "Which cluster does this stock belong to?"
   - "How does this compare to peers?"
   - "What factors drive the position?"

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'utils'"**
- Ensure you're running from the project root directory
- Check that `utils/__init__.py` exists

**"OpenAI API Error"**
- Verify your API key is correct in `.env`
- Check your OpenAI account has available credits
- The app will use fallback responses if API fails

**"Data not loaded"**
- Check if data files exist in `data/` directory
- Verify CSV file formats match expected columns
- Use `use_sample=True` for testing without real data

### VS Code Configuration

Recommended extensions:
- Python (Microsoft)
- Pylance
- Streamlit

Add to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "python.envFile": "${workspaceFolder}/.env"
}
```

## PCA Factor Loadings Reference

### PC1 (Quality/Stability) - 37.5% variance

| Factor | Loading | Interpretation |
|--------|---------|----------------|
| ROA | +0.44 | Quality |
| Earnings Yield | +0.39 | Value |
| ROE | +0.36 | Quality |
| Cash/Debt | +0.36 | Financial Strength |
| Momentum | +0.26 | Momentum |
| Book-to-Market | -0.35 | Value (inverse) |
| Volatility | -0.34 | Risk |

### PC2 (Size/Leverage) - 14.6% variance

| Factor | Loading | Interpretation |
|--------|---------|----------------|
| ADDV | +0.54 | Liquidity/Size |
| Debt-to-Assets | +0.48 | Leverage |
| Cash/Debt | -0.34 | Financial Strength |
| Sales-to-Price | -0.29 | Value |
| Gross Profitability | -0.27 | Quality |

## License

MIT License - feel free to use and modify for your own projects.

## Acknowledgments

- PCA methodology inspired by academic factor investing research
- Visualization design based on best practices for financial data
- Built with [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), and [OpenAI](https://openai.com/)

---

**Questions or Issues?** Open an issue on GitHub or contact the maintainer.
