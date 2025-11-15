import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

# ===== ML PIPELINE FEATURE ENGINEERING =====

def ml_pipeline_feature_engineering(df):
    """
    ML Pipeline feature engineering - stejný jako advanced_feature_engineering z streamlit_ml_app.py
    """
    if len(df) == 0:
        return df
    
    features = df.copy()
    
    # Základní rozklad kódů - stejný jako streamlit_ml_app.py
    features['balls_char1'] = features['balls_code'].astype(str).str[0]
    features['balls_mid'] = features['balls_code'].astype(str).str[1].apply(
        lambda x: ord(x) if not x.isdigit() else int(x)
    )
    features['balls_char2'] = features['balls_code'].astype(str).str[2]
    
    features['toy_num1'] = features['toy_code'].astype(str).str[:-2].astype(int)
    features['toy_char'] = features['toy_code'].astype(str).str[-2]
    features['toy_num2'] = features['toy_code'].astype(str).str[-1].astype(int)
    
    # Hash features pro výrobce - stejný jako streamlit_ml_app.py
    features['manufacturer_hash'] = (features['toy_num1'] % 3)
    features['batch_hash'] = (features['toy_num1'] // 8) % 8
    
    # Interakce mezi kódy - stejný jako streamlit_ml_app.py
    features['code_interaction'] = features['balls_mid'] * features['toy_num1']
    features['mod24_interaction'] = features['code_interaction'] % 24
    
    # Char kombinace - stejný jako streamlit_ml_app.py
    features['char_pattern'] = features['balls_char1'] + features['toy_char']
    
    # Location features
    features['has_location'] = features['location_state'].notna().astype(int)
    
    # DODATEČNÉ FEATURES PRO LEPŠÍ SEPARACI:
    # Cyclic features pro toy_num1 (24 hraček)
    features['toy_num1_mod24'] = features['toy_num1'] % 24
    features['toy_num1_div24'] = features['toy_num1'] // 24
    
    # Cyclic features pro balls_mid
    features['balls_mid_mod24'] = features['balls_mid'] % 24
    
    # Kombinované cyclic features
    features['toy_balls_cyclic'] = (features['toy_num1'] + features['balls_mid']) % 24
    
    # Manufacturer specific features (3 výrobci, každý 8 hraček)
    features['manufacturer_slot'] = features['toy_num1'] % 8  # slot v rámci výrobce
    features['toy_num1_mod8'] = features['toy_num1'] % 8
    
    # Power features pro toy_num1
    features['toy_num1_squared'] = features['toy_num1'] ** 2
    features['toy_num1_sqrt'] = np.sqrt(features['toy_num1'])
    
    # Advanced interactions
    features['manufacturer_batch_interaction'] = features['manufacturer_hash'] * 8 + features['batch_hash']
    features['code_interaction_mod24'] = (features['code_interaction']) % 24
    
    return features

def ml_pipeline_encode_features(features, categorical_encoders=None, return_encoders=False):
    """
    ML Pipeline encoding kategorických proměnných
    (z ml_pipeline.py)
    
    Args:
        features: DataFrame with features
        categorical_encoders: Dict of pre-fitted LabelEncoders for categorical columns (for prediction)
        return_encoders: If True, return the categorical encoders along with other outputs
    
    Returns:
        X_numeric.values: Encoded feature array
        y: Encoded target (or None)
        label_encoder: LabelEncoder for target (toy)
        feature_names: List of feature names
        categorical_encoders (if return_encoders=True): Dict of fitted categorical encoders
    """
    if len(features) == 0:
        if return_encoders:
            return None, None, None, None, None
        return None, None, None, None
    
    X = features.copy()
    label_encoder = LabelEncoder()
    
    # Encode kategorické proměnné - stejný seznam jako streamlit_ml_app.py
    categorical_cols = ['balls_char1', 'balls_char2', 'toy_char', 'char_pattern', 'location_state']
    
    # Store encoders if fitting new ones, or use provided encoders for prediction
    encoders_dict = categorical_encoders if categorical_encoders is not None else {}
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('NONE').astype(str)
            
            if col in encoders_dict:
                # Use pre-fitted encoder for prediction
                le = encoders_dict[col]
                # Handle unseen categories by assigning them a default value (max existing label + 1)
                # Or we can use transform with handling for unseen values
                try:
                    X[col] = le.transform(X[col])
                except ValueError:
                    # Handle unseen categories: assign max label + 1
                    unique_known = set(le.classes_)
                    unique_new = set(X[col].unique())
                    unseen = unique_new - unique_known
                    if unseen:
                        # For unseen categories, assign max encoded value + 1
                        max_label = len(le.classes_) - 1
                        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in unique_known else max_label + 1)
            else:
                # Fit new encoder for training
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders_dict[col] = le
    
    # Target encoding
    if 'toy' in X.columns:
        y = label_encoder.fit_transform(X['toy'])
    else:
        y = None
    
    # Drop unnecessary columns (including sequence features - they're artifacts)
    cols_to_drop = ['timestamp', 'balls_code', 'toy_code', 'toy', 'sequence_pos', 'sequence_mod24']
    X = X.drop([col for col in cols_to_drop if col in X.columns], axis=1)
    
    # Zajisti, že všechny sloupce jsou numeric
    for col in X.columns:
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            # Zkus převést na numeric, pokud to selže, zakóduj
            try:
                X[col] = pd.to_numeric(X[col], errors='raise')
            except (ValueError, TypeError):
                # Pokud převod selže, zakóduj jako kategorický
                if col in encoders_dict:
                    # Use existing encoder
                    le = encoders_dict[col]
                    try:
                        X[col] = le.transform(X[col].fillna('MISSING').astype(str))
                    except ValueError:
                        max_label = len(le.classes_) - 1
                        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else max_label + 1)
                else:
                    # Fit new encoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].fillna('MISSING').astype(str))
                    encoders_dict[col] = le
    
    feature_names = X.columns.tolist()
    
    # Zajisti, že X je numeric array
    X_numeric = X.select_dtypes(include=[np.number])
    if len(X_numeric.columns) != len(X.columns):
        # Některé sloupce nebyly numeric - zkus znovu převést
        for col in X.columns:
            if col not in X_numeric.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        X_numeric = X.select_dtypes(include=[np.number])
    
    if return_encoders:
        return X_numeric.values, y, label_encoder, X_numeric.columns.tolist(), encoders_dict
    return X_numeric.values, y, label_encoder, X_numeric.columns.tolist()

def ml_pipeline_clustering(X_scaled, n_clusters=24, y=None, toy_encoder=None, verbose=False):
    """
    ML Pipeline clustering analysis with detailed logging
    (z ml_pipeline.py)
    """
    if len(X_scaled) == 0:
        return None, None
    
    n_clusters = min(n_clusters, len(X_scaled))
    if n_clusters < 2:
        return None, None
    
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    if verbose and y is not None and toy_encoder is not None:
        # Log cluster analysis
        from datetime import datetime
        def log(msg):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")
        
        log("=== CLUSTERING ANALYSIS ===")
        sil_score = silhouette_score(X_scaled, clusters)
        log(f"Silhouette Score: {sil_score:.3f}")
        
        # Cluster-toy mapping
        toy_names = toy_encoder.inverse_transform(y)
        cluster_toy_mapping = pd.DataFrame({
            'cluster': clusters,
            'toy': toy_names
        })
        
        log("\n=== CLUSTER-TOY MAPPING ===")
        for cluster_id in range(n_clusters):
            cluster_data = cluster_toy_mapping[cluster_toy_mapping['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                toy_counts = cluster_data['toy'].value_counts()
                dominant_toy = toy_counts.index[0]
                dominant_count = toy_counts.iloc[0]
                purity = dominant_count / len(cluster_data)
                other_toys = ', '.join(toy_counts.index[1:3].tolist()[:2]) if len(toy_counts) > 1 else "none"
                log(f"Cluster {cluster_id:2d}: {len(cluster_data):3d} samples, "
                    f"dominant: {dominant_toy:20s} ({dominant_count}/{len(cluster_data)} = {purity:.1%}), "
                    f"other: {other_toys}")
        
        # Overall purity
        cluster_purities = []
        for cluster_id in range(n_clusters):
            cluster_data = cluster_toy_mapping[cluster_toy_mapping['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                toy_counts = cluster_data['toy'].value_counts()
                purity = toy_counts.iloc[0] / len(cluster_data)
                cluster_purities.append(purity)
        
        avg_purity = np.mean(cluster_purities) if cluster_purities else 0
        log(f"\nAverage cluster purity: {avg_purity:.1%}")
        log(f"Inertia: {kmeans.inertia_:.2f}")
    
    return clusters, kmeans

# ===== PCA ANALYSIS FUNCTIONS =====

def perform_pca_analysis(df_features, n_components=3, use_ml_pipeline=True):
    """
    Perform PCA analysis on numerical features - optimized for 24 toy prediction
    Uses ML Pipeline feature engineering with 3 manufacturer hypothesis
    
    Args:
        df_features: DataFrame with extracted features
        n_components: Number of principal components (default: 3)
        use_ml_pipeline: Use ML pipeline feature engineering (default: True)
    
    Returns:
        pca: Fitted PCA object
        pca_data: DataFrame with PCA components
        explained_variance: Explained variance ratio
        feature_names: List of feature names used
        X_scaled: Scaled feature matrix (for clustering)
        scaler: Fitted StandardScaler
        clusters: Cluster labels (24 clusters for 24 toys)
        toy_encoder: LabelEncoder for toys
        y: Encoded toy labels
    """
    if len(df_features) == 0:
        return None, None, None, None, None, None, None, None, None
    
    if use_ml_pipeline:
        # Use ML Pipeline feature engineering
        features = ml_pipeline_feature_engineering(df_features)
        X, y, toy_encoder, feature_names = ml_pipeline_encode_features(features, return_encoders=False)
        
        if X is None or len(X) == 0:
            return None, None, None, None, None, None, None, None, None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Always use 24 clusters for 24 toys
        n_clusters = 24
        if len(X_scaled) >= 24:
            clusters, kmeans_model = ml_pipeline_clustering(X_scaled, n_clusters=n_clusters, y=y, toy_encoder=toy_encoder, verbose=True)
        else:
            clusters = None
        
        # Perform PCA on scaled features (same approach as streamlit_ml_app.py)
        n_components = min(n_components, len(feature_names), len(X_scaled))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)
        
    else:
        return None, None, None, None, None, None, None, None, None
    
    # Create DataFrame with PCA components
    pca_data = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add toy information
    if y is not None and toy_encoder is not None:
        pca_data['toy'] = toy_encoder.inverse_transform(y)
        pca_data['toy_encoded'] = y
    elif 'toy' in df_features.columns:
        pca_data['toy'] = df_features['toy'].values
    
    # Add cluster information
    if clusters is not None:
        pca_data['cluster'] = clusters
    
    # Add original codes for hover data
    if 'balls_code' in df_features.columns:
        pca_data['balls_code'] = df_features['balls_code'].values
    if 'toy_code' in df_features.columns:
        pca_data['toy_code'] = df_features['toy_code'].values
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return pca, pca_data, explained_variance, feature_names, X_scaled, scaler, clusters, toy_encoder, y

def plot_pca_3d(pca_data, explained_variance, X_scaled=None, scaler=None, y=None, toy_encoder=None, color_by='toy'):
    """
    Create 3D PCA scatter plot - same approach as streamlit_ml_app.py
    Uses single trace with colorbar for continuous toy class coloring
    
    Args:
        pca_data: DataFrame with PCA components (PC1, PC2, PC3)
        explained_variance: Explained variance ratio for each component
        X_scaled: Scaled feature matrix (optional, for recalculating PCA)
        scaler: Fitted StandardScaler (optional)
        y: Encoded toy labels (required for same visualization)
        toy_encoder: LabelEncoder for toys (optional)
        color_by: 'toy' - always color by toy (24 toys = 24 classes)
    
    Returns:
        fig: Plotly 3D scatter plot figure with colorbar (same as streamlit_ml_app.py)
    """
    if pca_data is None or 'PC1' not in pca_data.columns or 'PC2' not in pca_data.columns:
        return None
    
    if 'PC3' not in pca_data.columns:
        return None
    
    # Get toy names
    if y is not None and toy_encoder is not None:
        toy_names = toy_encoder.inverse_transform(y)
    elif 'toy' in pca_data.columns:
        toy_names = pca_data['toy'].values
    else:
        return None
    
    # Add toy names to pca_data for grouping
    pca_data = pca_data.copy()
    pca_data['toy_name'] = toy_names
    
    # Get unique toys and assign colors
    unique_toys = sorted(pca_data['toy_name'].unique())
    
    # Use Plotly's qualitative color palette (enough colors for 24 toys)
    colors = px.colors.qualitative.Set3
    if len(unique_toys) > len(colors):
        # Extend palette if needed
        colors = colors * ((len(unique_toys) // len(colors)) + 1)
    
    # Create figure with one trace per toy
    fig = go.Figure()
    
    for i, toy in enumerate(unique_toys):
        toy_data = pca_data[pca_data['toy_name'] == toy]
        fig.add_trace(go.Scatter3d(
            x=toy_data['PC1'],
            y=toy_data['PC2'],
            z=toy_data['PC3'],
            mode='markers',
            name=toy,
            marker=dict(
                size=8,
                color=colors[i % len(colors)]
            ),
            hovertemplate=f'<b>{toy}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>'
        ))
    
    # Layout with legend at bottom
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({explained_variance[0]:.1%})',
            yaxis_title=f'PC2 ({explained_variance[1]:.1%})',
            zaxis_title=f'PC3 ({explained_variance[2]:.1%})',
            xaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            yaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            zaxis=dict(showbackground=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        margin=dict(l=0, r=0, b=100, t=0),  # Bottom margin for legend
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        uirevision='constant'
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
    # Increase bottom margin to accommodate rotated x-axis labels and text labels below them
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Loading',
        height=500,
        xaxis={'tickangle': -45, 'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        yaxis={'showgrid': True, 'gridcolor': 'rgba(128, 128, 128, 0.2)'},
        barmode='group',
        margin=dict(l=0, r=0, b=150, t=0),  # Increased bottom margin for rotated labels and text
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