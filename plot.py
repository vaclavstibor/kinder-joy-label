import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def extract_code_features(df):
    """
    Extract statistical features from balls_code and toy_code for predictive modeling
    """
    if len(df) == 0:
        return df
    
    df_features = df.copy()
    
    # Code length features
    df_features['balls_code_length'] = df_features['balls_code'].astype(str).str.len()
    df_features['toy_code_length'] = df_features['toy_code'].astype(str).str.len()
    
    # Numeric character count
    df_features['balls_code_numeric_count'] = df_features['balls_code'].astype(str).str.count(r'\d')
    df_features['toy_code_numeric_count'] = df_features['toy_code'].astype(str).str.count(r'\d')
    
    # Alphabetic character count
    df_features['balls_code_alpha_count'] = df_features['balls_code'].astype(str).str.count(r'[a-zA-Z]')
    df_features['toy_code_alpha_count'] = df_features['toy_code'].astype(str).str.count(r'[a-zA-Z]')
    
    # Uppercase count
    df_features['balls_code_upper_count'] = df_features['balls_code'].astype(str).str.count(r'[A-Z]')
    df_features['toy_code_upper_count'] = df_features['toy_code'].astype(str).str.count(r'[A-Z]')
    
    # Special character count
    df_features['balls_code_special_count'] = df_features['balls_code'].astype(str).str.count(r'[^a-zA-Z0-9]')
    df_features['toy_code_special_count'] = df_features['toy_code'].astype(str).str.count(r'[^a-zA-Z0-9]')
    
    # Sum of numeric values in code (if present)
    df_features['balls_code_numeric_sum'] = df_features['balls_code'].astype(str).apply(
        lambda x: sum([int(c) for c in x if c.isdigit()]) if any(c.isdigit() for c in x) else 0
    )
    df_features['toy_code_numeric_sum'] = df_features['toy_code'].astype(str).apply(
        lambda x: sum([int(c) for c in x if c.isdigit()]) if any(c.isdigit() for c in x) else 0
    )
    
    # First character features
    df_features['balls_code_first_char'] = df_features['balls_code'].astype(str).str[0]
    df_features['toy_code_first_char'] = df_features['toy_code'].astype(str).str[0]
    
    # Encode toy as numeric
    le = LabelEncoder()
    df_features['toy_encoded'] = le.fit_transform(df_features['toy'])
    
    return df_features

def calculate_code_statistics(df):
    """Calculate statistical measures for code patterns"""
    if len(df) == 0:
        return {}
    
    stats_dict = {
        'balls_code': {
            'mean_length': df['balls_code'].astype(str).str.len().mean(),
            'std_length': df['balls_code'].astype(str).str.len().std(),
            'unique_codes': df['balls_code'].nunique(),
            'duplicate_rate': 1 - (df['balls_code'].nunique() / len(df))
        },
        'toy_code': {
            'mean_length': df['toy_code'].astype(str).str.len().mean(),
            'std_length': df['toy_code'].astype(str).str.len().std(),
            'unique_codes': df['toy_code'].nunique(),
            'duplicate_rate': 1 - (df['toy_code'].nunique() / len(df))
        }
    }
    
    return stats_dict

# ===== PCA ANALYSIS FUNCTIONS =====

def perform_pca_analysis(df_features, n_components=3):
    """
    Perform PCA analysis on numerical features
    
    Args:
        df_features: DataFrame with extracted features
        n_components: Number of principal components (default: 3)
    
    Returns:
        pca: Fitted PCA object
        pca_data: DataFrame with PCA components
        explained_variance: Explained variance ratio
        feature_names: List of feature names used
    """
    if len(df_features) == 0:
        return None, None, None, None
    
    # Select numerical feature columns
    numeric_cols = [col for col in df_features.columns 
                   if col.endswith(('_length', '_count', '_sum', '_encoded')) 
                   and col != 'toy_encoded']
    
    if len(numeric_cols) < 2:
        return None, None, None, None
    
    # Extract numerical features
    X = df_features[numeric_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    n_components = min(n_components, len(numeric_cols), len(X_scaled))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA components
    pca_data = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add toy information
    pca_data['toy'] = df_features['toy'].values
    
    # Add original codes for hover data
    if 'balls_code' in df_features.columns:
        pca_data['balls_code'] = df_features['balls_code'].values
    if 'toy_code' in df_features.columns:
        pca_data['toy_code'] = df_features['toy_code'].values
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return pca, pca_data, explained_variance, numeric_cols

def plot_pca_3d(pca_data, explained_variance, color_by='toy', n_clusters=None):
    """
    Create 3D PCA scatter plot with clustering using viridis color scale
    
    Args:
        pca_data: DataFrame with PCA components (PC1, PC2, PC3)
        explained_variance: Explained variance ratio for each component
        color_by: 'toy' or 'cluster' - how to color points
        n_clusters: Number of clusters for KMeans (if color_by='cluster')
    
    Returns:
        fig: Plotly 3D scatter plot figure
    """
    if pca_data is None or 'PC1' not in pca_data.columns or 'PC2' not in pca_data.columns:
        return None
    
    # Check if we have PC3
    has_pc3 = 'PC3' in pca_data.columns
    if not has_pc3:
        return None
    
    # Prepare data copy
    pca_data = pca_data.copy()
    
    # Color by cluster if requested
    if color_by == 'cluster' and n_clusters is not None:
        # Perform KMeans clustering on PCA components
        n_clusters = min(n_clusters, len(pca_data))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(pca_data[['PC1', 'PC2', 'PC3']])
            pca_data['cluster'] = clusters
            color_column = 'cluster'
        else:
            color_column = 'toy'
    else:
        color_column = 'toy'
    
    # Create 3D scatter plot
    # For toy: use categorical colors with legend showing toy names
    # For cluster: use continuous viridis scale with colorbar
    if color_column == 'toy':
        # Get unique toys and assign viridis colors
        unique_toys = sorted(pca_data['toy'].unique())
        n_toys = len(unique_toys)
        
        # Get viridis colors - sample evenly across viridis scale
        viridis_palette = px.colors.sequential.Viridis  # 10 colors
        # Sample colors evenly - if 1 toy, use first color; if 2 toys, use first and last, etc.
        if n_toys == 1:
            viridis_colors = [viridis_palette[0]]
        else:
            # Sample evenly across the palette
            indices = [int(i * (len(viridis_palette) - 1) / (n_toys - 1)) for i in range(n_toys)]
            viridis_colors = [viridis_palette[idx] for idx in indices]
        
        # If more toys than colors in palette, repeat palette
        if n_toys > len(viridis_palette):
            viridis_colors = (viridis_colors * ((n_toys // len(viridis_colors)) + 1))[:n_toys]
        
        # Create figure with separate trace for each toy
        fig = go.Figure()
        for i, toy in enumerate(unique_toys):
            toy_data = pca_data[pca_data['toy'] == toy]
            fig.add_trace(go.Scatter3d(
                x=toy_data['PC1'],
                y=toy_data['PC2'],
                z=toy_data['PC3'],
                mode='markers',
                name=toy,
                marker=dict(
                    size=8,
                    color=viridis_colors[i],
                    opacity=0.7
                ),
                hovertemplate='<b>%{fullData.name}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
            ))
        
        layout_updates = {
            'legend': dict(
                x=1.02,
                y=0.02,
                xanchor='left',
                yanchor='bottom',
                bgcolor='rgba(0,0,0,0)',  # Transparent background
                bordercolor='rgba(0,0,0,0)',  # Transparent border
                borderwidth=0,
                font=dict(size=10)
            )
        }
    else:
        # Use continuous viridis scale for clusters
        fig = px.scatter_3d(
            pca_data,
            x='PC1',
            y='PC2',
            z='PC3',
            color=color_column,
            color_continuous_scale='viridis',
            hover_data=['toy', 'balls_code', 'toy_code'] if 'balls_code' in pca_data.columns else ['toy'],
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=8))
        
        # Update layout with colorbar in bottom right
        layout_updates = {
            'coloraxis_colorbar': dict(
                title=dict(text='Cluster', font=dict(size=10)),
                x=1.02,
                y=0.25,
                yanchor='bottom',
                len=0.5
            )
        }
    
    # Update layout - transparent theme
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            yaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            zaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        **layout_updates
    )
    
    return fig

def plot_pca_scatter(pca_data, explained_variance):
    """Create 2D PCA scatter plot colored by toy type (deprecated - use plot_pca_3d)"""
    if pca_data is None or 'PC1' not in pca_data.columns or 'PC2' not in pca_data.columns:
        return None
    
    fig = px.scatter(
        pca_data,
        x='PC1',
        y='PC2',
        color='toy',
        title=f'PCA Visualization (PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%} variance)',
        labels={
            'PC1': f'First Principal Component ({explained_variance[0]:.1%} variance)',
            'PC2': f'Second Principal Component ({explained_variance[1]:.1%} variance)'
        },
        hover_data=['toy']
    )
    
    fig.update_layout(height=600)
    return fig

def plot_pca_variance_explained(explained_variance):
    """Plot explained variance for each principal component with viridis theme"""
    if explained_variance is None:
        return None
    
    n_components = len(explained_variance)
    cumsum_variance = np.cumsum(explained_variance)
    
    df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(n_components)],
        'Individual Variance': explained_variance * 100,
        'Cumulative Variance': cumsum_variance * 100
    })
    
    fig = go.Figure()
    
    # Individual variance with viridis colors
    fig.add_trace(go.Bar(
        x=df['PC'],
        y=df['Individual Variance'],
        name='Individual Variance',
        marker=dict(
            color=df['Individual Variance'],
            colorscale='viridis',
            line=dict(color='rgb(8,48,107)', width=1.5)
        ),
        opacity=0.8
    ))
    
    # Cumulative variance
    fig.add_trace(go.Scatter(
        x=df['PC'],
        y=df['Cumulative Variance'],
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # Update layout - transparent theme
    fig.update_layout(
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance (%)',
        height=400,
        hovermode='x unified',
        xaxis={'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        yaxis={'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    
    return fig

def plot_pca_loadings(pca, feature_names, n_components=3):
    """Plot PCA loadings (component weights) for features with viridis theme"""
    if pca is None or feature_names is None:
        return None
    
    n_components = min(n_components, pca.n_components_)
    
    # Get loadings
    loadings = pca.components_[:n_components].T
    
    # Use viridis colors for different PCs
    colors = px.colors.sequential.Viridis[:n_components]
    if n_components > len(colors):
        colors = px.colors.sequential.Viridis * (n_components // len(px.colors.sequential.Viridis) + 1)
    
    fig = go.Figure()
    
    for i in range(n_components):
        fig.add_trace(go.Bar(
            x=feature_names,
            y=loadings[:, i],
            name=f'PC{i+1}',
            text=[f'{val:.2f}' for val in loadings[:, i]],
            textposition='outside',
        marker=dict(
                color=colors[i],
                line=dict(color=colors[i], width=1.5)
            ),
            opacity=0.8
        ))
    
    # Update layout - transparent theme
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Loading',
        height=500,
        xaxis={'tickangle': -45, 'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        yaxis={'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        barmode='group',
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ===== SIMPLE VISUALIZATION FUNCTIONS =====

def plot_toy_frequency_analysis(df):
    """Simple toy distribution chart with Streamlit red color"""
    if len(df) == 0:
        return None
    
    toy_counts = df['toy'].value_counts()
    mean_count = toy_counts.mean()
    
    # Create DataFrame for plotly express
    df_plot = pd.DataFrame({
        'Toy': toy_counts.index,
        'Frequency': toy_counts.values
    })
    
    # Use plotly express with Streamlit primary button color (red)
    streamlit_red = '#FF4B4B'  # Streamlit primary button color
    fig = px.bar(
        df_plot,
        x='Toy',
        y='Frequency',
        text='Frequency',
        labels={'Toy': 'Toy', 'Frequency': 'Frequency'}
    )
    
    # Update text position and set Streamlit red color with full opacity
    # Format text as integers and remove marker line
    fig.update_traces(
        textposition='outside',
        texttemplate='%{text:.0f}',
        text=df_plot['Frequency'].values,
        marker_color=streamlit_red,
        marker_line_width=0,
        opacity=1.0
    )
    
    # Add mean line
    fig.add_hline(
        y=mean_count, 
        line_dash="dash", 
        line_color="ghostWhite",
        annotation_text=f"Mean: {mean_count:.1f}",
        annotation_position="right"
    )
    
    # Update layout - transparent theme, only horizontal grid on integer values
    fig.update_layout(
        xaxis_title='Toy',
        yaxis_title='Frequency',
        height=500,
        xaxis={'tickangle': -45, 'showgrid': False},
        yaxis={
            'showgrid': True, 
            'gridcolor': 'rgba(128, 128, 128, 0.2)',  # Subtle gray grid, less visible
            'dtick': 1,  # Only show grid lines at integer values
            'tickmode': 'linear'
        },
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
