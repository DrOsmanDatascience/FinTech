"""
Visualization Utilities
=======================
Functions for creating interactive PCA visualizations using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# =============================================================================
# COLOR SCHEMES & STYLING
# =============================================================================

# Professional color palette
COLORS = {
    'cluster_0': '#667eea',  # High Quality / Large-Liquid (purple-blue)
    'cluster_1': '#f6ad55',  # Lower Quality / Large-Liquid (orange)
    'cluster_2': '#48bb78',  # High Quality / Cash-Rich (green)
    'cluster_3': '#fc8181',  # Lower Quality / Smaller (red)
    'selected': '#1a1a2e',   # Selected stock (dark)
    'grid': '#e2e8f0',       # Grid lines
    'text': '#2d3748',       # Text color
    'background': '#ffffff'   # Background
}

CLUSTER_NAMES = {
    0: 'High Quality / Large-Liquid',
    1: 'Lower Quality / Large-Liquid',
    2: 'High Quality / Cash-Rich',
    3: 'Lower Quality / Smaller'
}

# =============================================================================
# MAIN PCA SCATTER PLOT
# =============================================================================

def create_pca_scatter(pca_data, selected_stock=None):
    """
    Create an interactive PCA scatter plot showing all stocks.
    
    Parameters:
    -----------
    pca_data : pd.DataFrame
        DataFrame with columns: ticker, company_name, pc1, pc2, cluster
    selected_stock : dict or None
        Dictionary containing selected stock information
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive scatter plot
    """
    
    # Create color mapping for clusters
    pca_data['cluster_name'] = pca_data['cluster'].map(CLUSTER_NAMES)
    color_map = {
        CLUSTER_NAMES[0]: COLORS['cluster_0'],
        CLUSTER_NAMES[1]: COLORS['cluster_1'],
        CLUSTER_NAMES[2]: COLORS['cluster_2'],
        CLUSTER_NAMES[3]: COLORS['cluster_3']
    }
    
    # Create base scatter plot
    fig = px.scatter(
        pca_data,
        x='pc1',
        y='pc2',
        color='cluster_name',
        color_discrete_map=color_map,
        hover_name='ticker',
        hover_data={
            'company_name': True,
            'pc1': ':.2f',
            'pc2': ':.2f',
            'market_weight': ':.2f',
            'cluster_name': True
        },
        labels={
            'pc1': 'PC1: Quality / Stability →',
            'pc2': 'PC2: Size / Leverage →',
            'cluster_name': 'Cluster'
        },
        title='Stock Positions in PCA Space'
    )
    
    # Update marker styling
    fig.update_traces(
        marker=dict(
            size=10,
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        selector=dict(mode='markers')
    )
    
    # Add quadrant dividing lines
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['grid'], line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['grid'], line_width=1)
    
    # Add quadrant labels as annotations
    x_range = pca_data['pc1'].max() - pca_data['pc1'].min()
    y_range = pca_data['pc2'].max() - pca_data['pc2'].min()
    
    annotations = [
        # Top-right: High Quality / Large-Liquid
        dict(
            x=pca_data['pc1'].max() - x_range * 0.15,
            y=pca_data['pc2'].max() - y_range * 0.1,
            text="<b>High-Quality</b><br>Large/Liquid",
            showarrow=False,
            font=dict(size=10, color=COLORS['cluster_0']),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        ),
        # Top-left: Lower Quality / Large-Liquid
        dict(
            x=pca_data['pc1'].min() + x_range * 0.15,
            y=pca_data['pc2'].max() - y_range * 0.1,
            text="<b>Lower-Quality</b><br>Large/Liquid",
            showarrow=False,
            font=dict(size=10, color=COLORS['cluster_1']),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        ),
        # Bottom-right: High Quality / Cash-Rich
        dict(
            x=pca_data['pc1'].max() - x_range * 0.15,
            y=pca_data['pc2'].min() + y_range * 0.1,
            text="<b>High-Quality</b><br>Cash-Rich",
            showarrow=False,
            font=dict(size=10, color=COLORS['cluster_2']),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        ),
        # Bottom-left: Lower Quality / Smaller
        dict(
            x=pca_data['pc1'].min() + x_range * 0.15,
            y=pca_data['pc2'].min() + y_range * 0.1,
            text="<b>Lower-Quality</b><br>Cash-Rich",
            showarrow=False,
            font=dict(size=10, color=COLORS['cluster_3']),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4
        )
    ]
    
    # Highlight selected stock if provided
    if selected_stock:
        ticker = selected_stock['ticker']
        pc1 = selected_stock['pc1']
        pc2 = selected_stock['pc2']
        
        # Add highlighted marker for selected stock
        fig.add_trace(
            go.Scatter(
                x=[pc1],
                y=[pc2],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=COLORS['selected'],
                    line=dict(width=3, color='white'),
                    symbol='diamond'
                ),
                text=[ticker],
                textposition='top center',
                textfont=dict(size=14, color=COLORS['selected'], family='JetBrains Mono'),
                name=f'Selected: {ticker}',
                hovertemplate=(
                    f"<b>{ticker}</b><br>"
                    f"{selected_stock.get('company_name', '')}<br>"
                    f"PC1: {pc1:.2f}<br>"
                    f"PC2: {pc2:.2f}<br>"
                    f"Market Weight: {selected_stock.get('market_weight', 0):.2f}%"
                    "<extra></extra>"
                )
            )
        )
        
        # Add annotation pointing to selected stock
        annotations.append(
            dict(
                x=pc1,
                y=pc2,
                xref='x',
                yref='y',
                text=f"<b>{ticker}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLORS['selected'],
                ax=40,
                ay=-40,
                font=dict(size=12, color=COLORS['selected']),
                bgcolor='white',
                bordercolor=COLORS['selected'],
                borderwidth=2,
                borderpad=4
            )
        )
    
    # Update layout
    fig.update_layout(
        annotations=annotations,
        font=dict(family='DM Sans', size=12),
        title=dict(
            text='<b>Stock Positions in PCA Space</b><br>'
                 '<span style="font-size:12px;color:#666;">PC1 + PC2 explain 52.5% of total variance</span>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='<b>PC1: Quality / Stability →</b><br>'
                  '<span style="font-size:10px;">Higher = More profitable, financially strong</span>',
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid'],
            showgrid=True
        ),
        yaxis=dict(
            title='<b>PC2: Size / Leverage →</b><br>'
                  '<span style="font-size:10px;">Higher = Larger, more leveraged</span>',
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid'],
            showgrid=True
        ),
        legend=dict(
            title='<b>Cluster</b>',
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        margin=dict(l=80, r=40, t=100, b=120)
    )
    
    return fig


# =============================================================================
# QUADRANT ANALYSIS PLOT
# =============================================================================

def create_quadrant_analysis(pca_data, selected_stock):
    """
    Create a plot focusing on the selected stock's quadrant with peer comparison.
    
    Parameters:
    -----------
    pca_data : pd.DataFrame
        Full PCA dataset
    selected_stock : dict
        Selected stock information
    
    Returns:
    --------
    tuple (plotly.Figure, pd.DataFrame)
        (Quadrant scatter plot, DataFrame of peer stocks)
    """
    
    ticker = selected_stock['ticker']
    cluster = selected_stock['cluster']
    pc1 = selected_stock['pc1']
    pc2 = selected_stock['pc2']
    
    # Filter to same quadrant
    quadrant_data = pca_data[pca_data['cluster'] == cluster].copy()
    peer_data = quadrant_data[quadrant_data['ticker'] != ticker]
    
    # Get cluster color
    cluster_color = COLORS[f'cluster_{cluster}']
    
    # Create figure
    fig = go.Figure()
    
    # Add peer stocks (hollow circles)
    fig.add_trace(
        go.Scatter(
            x=peer_data['pc1'],
            y=peer_data['pc2'],
            mode='markers',
            marker=dict(
                size=12,
                color='white',
                line=dict(width=2, color=cluster_color),
                opacity=0.8
            ),
            text=peer_data['ticker'],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "PC1: %{x:.2f}<br>"
                "PC2: %{y:.2f}<br>"
                "<extra></extra>"
            ),
            name='Quadrant Peers'
        )
    )
    
    # Add selected stock (filled with label)
    fig.add_trace(
        go.Scatter(
            x=[pc1],
            y=[pc2],
            mode='markers+text',
            marker=dict(
                size=25,
                color=cluster_color,
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=[ticker],
            textposition='top center',
            textfont=dict(size=14, color=COLORS['selected'], family='JetBrains Mono', weight='bold'),
            name=f'Selected: {ticker}',
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"PC1: {pc1:.2f}<br>"
                f"PC2: {pc2:.2f}<br>"
                f"Market Weight: {selected_stock.get('market_weight', 0):.2f}%<br>"
                "<extra></extra>"
            )
        )
    )
    
    # Add quadrant boundary box
    x_min, x_max = quadrant_data['pc1'].min() - 0.5, quadrant_data['pc1'].max() + 0.5
    y_min, y_max = quadrant_data['pc2'].min() - 0.5, quadrant_data['pc2'].max() + 0.5
    
    # Draw dashed boundary
    fig.add_shape(
        type='rect',
        x0=0 if pc1 >= 0 else x_min,
        y0=0 if pc2 >= 0 else y_min,
        x1=x_max if pc1 >= 0 else 0,
        y1=y_max if pc2 >= 0 else 0,
        line=dict(color=cluster_color, width=2, dash='dash'),
        fillcolor=f'rgba({int(cluster_color[1:3], 16)}, {int(cluster_color[3:5], 16)}, {int(cluster_color[5:7], 16)}, 0.05)'
    )
    
    # Update layout
    quadrant_name = CLUSTER_NAMES[cluster]
    fig.update_layout(
        title=dict(
            text=f'<b>Quadrant Analysis: {quadrant_name}</b><br>'
                 f'<span style="font-size:12px;color:#666;">{len(quadrant_data)} stocks in this quadrant</span>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='PC1: Quality / Stability →',
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='PC2: Size / Leverage →',
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid']
        ),
        font=dict(family='DM Sans'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        height=500,
        margin=dict(l=60, r=40, t=100, b=80)
    )
    
    return fig, quadrant_data


# =============================================================================
# TIME-LAPSE ANIMATION
# =============================================================================

def create_time_lapse_animation(pca_data, ticker):
    """
    Create an animated plot showing stock movement over time.
    
    Parameters:
    -----------
    pca_data : pd.DataFrame
        Full PCA dataset (for reference positions)
    ticker : str
        Stock ticker to animate
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Animated scatter plot with play controls
    """
    from utils.data_loader import load_time_series_pca
    
    # Load time series data
    ts_data = load_time_series_pca(ticker, use_sample=True)
    
    if ts_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No historical data available for {ticker}",
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get current stock info
    current_stock = pca_data[pca_data['ticker'] == ticker].iloc[0]
    cluster = current_stock['cluster']
    cluster_color = COLORS[f'cluster_{cluster}']
    
    # Create figure with frames for animation
    fig = go.Figure()
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['grid'], line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['grid'], line_width=1)
    
    # Add trail (all historical positions, faded)
    fig.add_trace(
        go.Scatter(
            x=ts_data['pc1'],
            y=ts_data['pc2'],
            mode='lines+markers',
            line=dict(color=cluster_color, width=1, dash='dot'),
            marker=dict(
                size=6,
                color=cluster_color,
                opacity=0.3,
                symbol='circle'
            ),
            name='Historical Path',
            hoverinfo='skip'
        )
    )
    
    # Create animation frames
    frames = []
    for i in range(len(ts_data)):
        frame_data = ts_data.iloc[:i+1]
        current_point = ts_data.iloc[i]
        
        frame = go.Frame(
            data=[
                # Trail up to current point
                go.Scatter(
                    x=frame_data['pc1'],
                    y=frame_data['pc2'],
                    mode='lines+markers',
                    line=dict(color=cluster_color, width=2),
                    marker=dict(
                        size=[6]*len(frame_data),
                        color=cluster_color,
                        opacity=[0.3 + 0.7*(j/(len(frame_data))) for j in range(len(frame_data))]
                    ),
                    showlegend=False
                ),
                # Current position (large marker)
                go.Scatter(
                    x=[current_point['pc1']],
                    y=[current_point['pc2']],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color=cluster_color,
                        line=dict(width=3, color='white'),
                        symbol='diamond'
                    ),
                    text=[ticker],
                    textposition='top center',
                    textfont=dict(size=14, family='JetBrains Mono', color=COLORS['selected']),
                    showlegend=False
                )
            ],
            name=str(current_point['date'].strftime('%Y-%m'))
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add current position marker
    current_point = ts_data.iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[current_point['pc1']],
            y=[current_point['pc2']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=cluster_color,
                line=dict(width=3, color='white'),
                symbol='diamond'
            ),
            text=[ticker],
            textposition='top center',
            textfont=dict(size=14, family='JetBrains Mono', color=COLORS['selected']),
            name=f'Current: {ticker}'
        )
    )
    
    # Add play/pause buttons and slider
    fig.update_layout(
        title=dict(
            text=f'<b>Historical Position: {ticker}</b><br>'
                 '<span style="font-size:12px;color:#666;">Movement over past 24 months</span>',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='PC1: Quality / Stability →',
            range=[ts_data['pc1'].min() - 1, ts_data['pc1'].max() + 1],
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title='PC2: Size / Leverage →',
            range=[ts_data['pc2'].min() - 0.5, ts_data['pc2'].max() + 0.5],
            zeroline=True,
            zerolinecolor=COLORS['grid'],
            gridcolor=COLORS['grid']
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.1,
                xanchor='left',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )
                        ]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate'
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=len(frames) - 1,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='Date: ',
                    visible=True,
                    xanchor='center'
                ),
                len=0.9,
                x=0.05,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                        label=f.name,
                        method='animate'
                    ) for f in frames
                ]
            )
        ],
        font=dict(family='DM Sans'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=550,
        margin=dict(l=60, r=40, t=120, b=100)
    )
    
    return fig


# =============================================================================
# FACTOR BREAKDOWN CHARTS
# =============================================================================

def create_factor_breakdown_chart(selected_stock, pca_data):
    """
    Create radar and bar charts showing factor scores.
    
    Parameters:
    -----------
    selected_stock : dict
        Selected stock information with factor scores
    pca_data : pd.DataFrame
        Full PCA data for percentile calculations
    
    Returns:
    --------
    tuple (plotly.Figure, plotly.Figure)
        (Radar chart, Bar chart)
    """
    
    # Factor names and scores
    factors = ['Value', 'Quality', 'Fin. Strength', 'Momentum', 'Volatility', 'Liquidity']
    factor_keys = ['value_score', 'quality_score', 'fin_strength_score', 
                   'momentum_score', 'volatility_score', 'liquidity_score']
    
    scores = [selected_stock.get(k, 0.5) for k in factor_keys]
    
    # Get cluster peers for comparison
    cluster = selected_stock['cluster']
    cluster_data = pca_data[pca_data['cluster'] == cluster]
    
    # Calculate percentiles within cluster
    percentiles = []
    for key in factor_keys:
        if key in cluster_data.columns:
            stock_value = selected_stock.get(key, 0.5)
            pctl = (cluster_data[key] < stock_value).mean() * 100
            percentiles.append(pctl)
        else:
            percentiles.append(50)
    
    cluster_color = COLORS[f'cluster_{cluster}']
    
    # ----- RADAR CHART -----
    fig_radar = go.Figure()
    
    # Add cluster average for comparison
    cluster_avg = [cluster_data[k].mean() if k in cluster_data.columns else 0.5 for k in factor_keys]
    
    fig_radar.add_trace(
        go.Scatterpolar(
            r=cluster_avg + [cluster_avg[0]],  # Close the polygon
            theta=factors + [factors[0]],
            fill='toself',
            fillcolor=f'rgba({int(cluster_color[1:3], 16)}, {int(cluster_color[3:5], 16)}, {int(cluster_color[5:7], 16)}, 0.2)',
            line=dict(color=cluster_color, width=1, dash='dash'),
            name='Cluster Average'
        )
    )
    
    fig_radar.add_trace(
        go.Scatterpolar(
            r=scores + [scores[0]],  # Close the polygon
            theta=factors + [factors[0]],
            fill='toself',
            fillcolor=f'rgba({int(cluster_color[1:3], 16)}, {int(cluster_color[3:5], 16)}, {int(cluster_color[5:7], 16)}, 0.5)',
            line=dict(color=cluster_color, width=2),
            name=selected_stock['ticker']
        )
    )
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1],
                ticktext=['25%', '50%', '75%', '100%']
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        font=dict(family='DM Sans'),
        margin=dict(l=60, r=60, t=40, b=60),
        height=350
    )
    
    # ----- BAR CHART (Percentiles) -----
    fig_bar = go.Figure()
    
    # Color bars based on percentile (gradient)
    bar_colors = [
        '#fc8181' if p < 25 else '#f6ad55' if p < 50 else '#48bb78' if p < 75 else '#667eea'
        for p in percentiles
    ]
    
    fig_bar.add_trace(
        go.Bar(
            x=factors,
            y=percentiles,
            marker=dict(
                color=bar_colors,
                line=dict(width=1, color='white')
            ),
            text=[f'{p:.0f}%' for p in percentiles],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Percentile: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Add median line
    fig_bar.add_hline(y=50, line_dash="dash", line_color=COLORS['grid'], 
                       annotation_text="Median", annotation_position="right")
    
    fig_bar.update_layout(
        title=dict(
            text='<b>Percentile Rank vs Cluster Peers</b>',
            font=dict(size=14)
        ),
        yaxis=dict(
            title='Percentile',
            range=[0, 105],
            tickvals=[0, 25, 50, 75, 100],
            gridcolor=COLORS['grid']
        ),
        xaxis=dict(
            title='',
            tickangle=-45
        ),
        font=dict(family='DM Sans'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=80),
        height=350
    )
    
    return fig_radar, fig_bar


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_quadrant_description(cluster):
    """Get detailed description for a cluster/quadrant."""
    descriptions = {
        0: {
            'name': 'High Quality / Large-Liquid',
            'pc1_desc': 'High Quality/Stability (PC1 > 0)',
            'pc2_desc': 'Large/Leveraged (PC2 > 0)',
            'characteristics': [
                'Higher profitability (ROA, ROE)',
                'Strong earnings yield',
                'Good cash/debt ratios',
                'Positive momentum',
                'Lower volatility',
                'Higher trading volume/liquidity',
                'More leveraged capital structure'
            ]
        },
        1: {
            'name': 'Lower Quality / Large-Liquid',
            'pc1_desc': 'Lower Quality/Stability (PC1 < 0)',
            'pc2_desc': 'Large/Leveraged (PC2 > 0)',
            'characteristics': [
                'Lower profitability metrics',
                'Higher volatility',
                'Growth-oriented characteristics',
                'High trading volume/liquidity',
                'More leveraged capital structure'
            ]
        },
        2: {
            'name': 'High Quality / Cash-Rich',
            'pc1_desc': 'High Quality/Stability (PC1 > 0)',
            'pc2_desc': 'Cash-Rich (PC2 < 0)',
            'characteristics': [
                'Higher profitability (ROA, ROE)',
                'Strong balance sheets',
                'Lower leverage / more cash',
                'Operationally efficient',
                'Lower volatility'
            ]
        },
        3: {
            'name': 'Lower Quality / Cash-Rich',
            'pc1_desc': 'Lower Quality/Stability (PC1 < 0)',
            'pc2_desc': 'Cash-Rich (PC2 < 0)',
            'characteristics': [
                'Lower profitability metrics',
                'Higher volatility',
                'Smaller/less liquid',
                'Lower leverage',
                'More speculative'
            ]
        }
    }
    return descriptions.get(cluster, descriptions[0])


if __name__ == "__main__":
    # Test visualizations
    from data_loader import load_pca_data, validate_ticker
    
    print("Testing visualizations...")
    
    pca_data = load_pca_data(use_sample=True)
    
    # Test PCA scatter
    fig = create_pca_scatter(pca_data)
    print("PCA scatter created successfully")
    
    # Test with selected stock
    is_valid, stock = validate_ticker("CL", pca_data, "ticker")
    if is_valid:
        fig = create_pca_scatter(pca_data, selected_stock=stock)
        print(f"PCA scatter with {stock['ticker']} highlighted")
        
        # Test quadrant analysis
        fig_quad, peers = create_quadrant_analysis(pca_data, stock)
        print(f"Quadrant analysis: {len(peers)} peers found")
        
        # Test time lapse
        fig_anim = create_time_lapse_animation(pca_data, stock['ticker'])
        print("Time lapse animation created")
        
        # Test factor breakdown
        fig_radar, fig_bar = create_factor_breakdown_chart(stock, pca_data)
        print("Factor breakdown charts created")
    
    print("\nAll visualization tests passed!")
