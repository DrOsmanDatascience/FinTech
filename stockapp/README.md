# Stock PCA Cluster Analysis - Streamlit Web Application

A comprehensive interactive dashboard for analyzing stock clusters using Principal Component Analysis (PCA). This application visualizes stock characteristics across quality, stability, leverage, and size dimensions.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

### Interactive Visualizations
- **2D PCA Scatter Plot**: Main visualization showing stocks positioned by quality/stability (PC1) and size/leverage (PC2)
- **Quadrant Analysis**: Peer comparison within the same market quadrant
- **Factor Breakdown**: Radar charts and percentile rankings for detailed factor analysis
- **Time-Lapse Animation**: Historical movement of stocks in PCA space
- **3D Visualization**: Interactive 3D view of cluster positions

### AI-Powered Chatbot
- Context-aware responses about cluster analysis
- Stock-specific insights and comparisons
- Powered by OpenAI's GPT models

## 📊 PCA Axis Interpretations

### PC1 (X-axis): Quality / Stability / Balance-Sheet Strength
- **High values (→ Right)**: High-quality, profitable, financially strong
- **Low values (← Left)**: Riskier, lower-quality, volatile
- **Explains ~37.5% of variance**

### PC2 (Y-axis): Size / Leverage / Capital Structure
- **High values (↑ Up)**: Large/liquid, leveraged
- **Low values (↓ Down)**: Cash-rich, operationally efficient
- **Explains ~14.6% of variance**

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- VS Code (recommended) or any Python IDE
- OpenAI API key (optional, for chatbot feature)

### Installation

1. **Clone or download the project**
   ```bash
   cd stock_pca_app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys** (optional, for chatbot)
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```
   
   Or enter the API key in the sidebar when running the app.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
stock_pca_app/
├── app.py              # Main Streamlit application
├── config.py           # Configuration and constants
├── utils.py            # Data processing and analysis utilities
├── visualizations.py   # Plotly visualization functions
├── chatbot.py          # OpenAI chatbot integration
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── data/               # Local data storage (optional)
    └── factors_build_df_dec_10.csv
```

## 🔧 Configuration

### Data Source
The application loads data from GitHub by default:
```python
# config.py
GITHUB_DATA_URL = "https://raw.githubusercontent.com/DrOsmanDatascience/FinTech/main/data/factors_build_df_dec_10.csv"
```

To use local data, place your CSV file in the `data/` folder and update `config.py`:
```python
LOCAL_DATA_PATH = "data/factors_build_df_dec_10.csv"
```

### Customize Clustering
Modify these settings in `config.py`:
```python
N_COMPONENTS = 3  # Number of PCA components
N_CLUSTERS = 4    # Number of KMeans clusters
```

### OpenAI Model
Change the chatbot model in `config.py`:
```python
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective
# or
OPENAI_MODEL = "gpt-4o"       # Higher quality
```

## 📈 Usage

### Basic Analysis
1. Enter a stock ticker (e.g., `AAPL`, `MSFT`) or PERMNO in the sidebar
2. View the stock's position in the PCA cluster plot
3. Explore different visualization tabs

### Chatbot Questions
Ask questions like:
- "Which cluster does this stock belong to?"
- "How does this stock compare to others in its cluster?"
- "What does the PC1 score tell me about this stock?"
- "Is this stock considered high quality or risky?"

## 🔒 Security Notes

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` files or secure environment variable management
- **Data Privacy**: The application processes data locally in your browser session

## 📝 Data Requirements

The CSV file should contain these columns:
- `permno`: CRSP permanent security identifier
- `ticker`: Stock ticker symbol
- `public_date` or `date`: Date column
- Factor columns: `earnings_yield`, `bm`, `sales_to_price`, `roe`, `roa`, `gprof`, `debt_assets`, `cash_debt`, `momentum_12m`, `vol_60d_ann`, `addv_63d`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Data sourced from CRSP via the FinTech repository
- Built with Streamlit, Plotly, and scikit-learn
- AI features powered by OpenAI

## 📞 Support

For questions or issues:
1. Check the existing documentation
2. Open an issue on GitHub
3. Contact the development team

---

**Happy Analyzing! 📊📈**
