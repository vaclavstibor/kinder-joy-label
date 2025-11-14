import streamlit as st
import pandas as pd
from datetime import datetime

# Import vizualizační modul
from plot import (
    plot_toy_frequency_analysis,
    extract_code_features,
    perform_pca_analysis,
    plot_pca_3d,
    plot_pca_loadings
)

# Import Firestore - required for the app to work
try:
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    from google.oauth2 import service_account
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Kinder Joy Labeling App",
    layout="wide"
)

# Check if Firestore is available after page config
if not FIRESTORE_AVAILABLE:
    st.error("Firestore packages not installed. Please install google-cloud-firestore and google-auth.")
    st.stop()

# Toy list (24 toys)
TOYS = [
    "Nancy", "Mike", "Lucas", "Demogordon", "Steve", "Eleven", "Vecna", 
    "Eleven Down", "Max Down", "Eleven clip", "Demogordon pen", 
    "Steven and Robin pen", "Erica kabel", "Demogordon clip", "Will", "Max", 
    "Dustin", "Hopper", "Will Down", "Steve Down", "Eddie Down", 
    "Dustin Down", "Hopper Down", "Robin Down"
]

# Firestore configuration
COLLECTION_NAME = "labels"  # Collection name in Firestore - each label is a separate document

def get_firestore_client():
    """Initialize Firestore client using Streamlit secrets (from secrets.toml locally or Streamlit Cloud secrets)"""
    if not FIRESTORE_AVAILABLE:
        return None
    
    try:
        # Check if credentials are in Streamlit secrets (works with secrets.toml locally or Streamlit Cloud secrets)
        if 'gcp_service_account' in st.secrets:
            # Convert Streamlit secrets to dict - it behaves like a dict
            creds_dict = dict(st.secrets['gcp_service_account'])
            
            # Fix private_key: replace \n escape sequences with actual newlines
            if 'private_key' in creds_dict and isinstance(creds_dict['private_key'], str):
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            db = firestore.Client(credentials=credentials, project=creds_dict.get('project_id'))
            return db
    except Exception as e:
        st.error(f"Firestore not configured: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None
    return None

@st.cache_data(ttl=5, show_spinner=False)  # Cache for 5 seconds - balances freshness with performance on Streamlit Cloud
def load_existing_data():
    """Load existing labeled data from Firestore"""
    db = get_firestore_client()
    if not db:
        st.error("Firestore is not configured. Please set up your credentials in Streamlit Secrets.")
        return pd.DataFrame(columns=["timestamp", "balls_code", "toy_code", "toy", "location_state"])
    
    try:
        # Get all documents from the collection
        docs = db.collection(COLLECTION_NAME).stream()
        
        # Convert to list of dictionaries
        records = []
        for doc in docs:
            doc_data = doc.to_dict()
            if doc_data:  # Only add non-empty documents
                records.append(doc_data)
        
        if records:
            df = pd.DataFrame(records)
            # Ensure all required columns exist
            required_columns = ["timestamp", "balls_code", "toy_code", "toy", "location_state"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Filter out empty rows
            df = df.dropna(how='all')
            df = df[~(df.astype(str).apply(lambda x: x.str.strip().eq('')).all(axis=1))]
            return df[required_columns]  # Return only required columns in correct order
        else:
            return pd.DataFrame(columns=["timestamp", "balls_code", "toy_code", "toy", "location_state"])
    except Exception as e:
        st.error(f"Error loading from Firestore: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "balls_code", "toy_code", "toy", "location_state"])

def save_backup(new_row):
    """Save individual submitted item to Firestore"""
    db = get_firestore_client()
    if not db:
        st.error("Firestore is not configured. Please set up your credentials in Streamlit Secrets.")
        return False
    
    try:
        # Check for duplicates before adding
        # Query for existing document with same data
        existing_docs = db.collection(COLLECTION_NAME)\
            .where(filter=FieldFilter('timestamp', '==', new_row['timestamp']))\
            .where(filter=FieldFilter('balls_code', '==', new_row['balls_code']))\
            .where(filter=FieldFilter('toy_code', '==', new_row['toy_code']))\
            .limit(1)\
            .stream()
        
        # Check if duplicate exists
        is_duplicate = any(True for _ in existing_docs)
        
        if not is_duplicate:
            # Create a new document with auto-generated ID
            # This avoids write conflicts - each write is to a different document
            db.collection(COLLECTION_NAME).add(new_row)
            return True  # Successfully saved to Firestore
        else:
            # Duplicate found, skip saving
            st.info("This label already exists, skipping duplicate.")
            return False
    except Exception as e:
        st.error(f"Error saving to Firestore: {str(e)}")
        return False

def save_data(df):
    """Save DataFrame to Firestore (each row as separate document)"""
    # Filter out empty rows before saving
    df = df.dropna(how='all')
    df = df[~(df.astype(str).apply(lambda x: x.str.strip().eq('')).all(axis=1))]
    
    # Remove duplicates based on all columns to prevent duplicate entries
    df = df.drop_duplicates(subset=["timestamp", "toy", "balls_code", "toy_code", "location_state"], keep='last')
    
    # For Firestore, individual saves are handled in save_backup()
    # Each label is saved as a separate document to avoid write conflicts
    # This function is kept for compatibility but doesn't do anything for Firestore
    pass

def main():
    
    # Centered title
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 2rem;'>Stranger Things Dataset Labeling</h1>",
        unsafe_allow_html=True
    )

    # Description
    st.markdown(
        "<p style='text-align: center; margin-bottom: 2rem;'>This app is used to make labeled Stranger Things Kinder Joy toys dataset which will be used to make statistical predictive model.</p>",
        unsafe_allow_html=True
    )

    
    # Load existing data (cached, so it's fast)
    df = load_existing_data()
    
    # Main form
    with st.form("labeling_form", clear_on_submit=True):
        # All inputs in one row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 1. Balls code
            balls_code = st.text_input(
                "Balls Code *",
                help="Must be at least 1 character",
                placeholder="Enter balls code"
            )
        
        with col2:
            # 2. Toy code
            toy_code = st.text_input(
                "Toy Code *",
                help="Must be at least 1 character",
                placeholder="Enter toy code"
            )
        
        with col3:
            # 3. Toy selection
            selected_toy = st.selectbox(
                "Select Toy *",
                options=[""] + TOYS,
                help="Select one of the 24 available toys"
            )
        
        with col4:
            # 4. Location state (optional)
            location_state = st.text_input(
                "Location State (Optional)",
                help="Optional location information",
                placeholder="Enter location state"
            )
        
        # Wide Submit button below inputs
        submitted = st.form_submit_button("Submit", type="primary", width='stretch')
        
        if submitted:
            # Validation
            errors = []
            
            if not selected_toy or selected_toy == "":
                errors.append("Please select a toy")
            
            if not balls_code or len(balls_code.strip()) < 1:
                errors.append("Balls code must be at least 1 character")
            
            if not toy_code or len(toy_code.strip()) < 1:
                errors.append("Toy code must be at least 1 character")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Create new row
                new_row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "toy": selected_toy,
                    "balls_code": balls_code.strip(),
                    "toy_code": toy_code.strip(),
                    "location_state": location_state.strip() if location_state else ""
                }
                
                # Save with status indicator (non-blocking)
                with st.status("Saving label...", expanded=False) as status:
                    # Reload latest data to get any concurrent submissions from other users
                    load_existing_data.clear()  # Clear cache first to get fresh data
                    
                    # Save new row to Firestore
                    if save_backup(new_row):
                        status.update(label="Label saved!", state="complete")
                    else:
                        status.update(label="Failed to save label", state="error")
                        st.stop()
                
                # Clear cache to force reload on next render so all users see the update
                load_existing_data.clear()
                
                # Reload to get updated count
                df_final = load_existing_data()
                st.success(f"Label saved successfully! (Total: {len(df_final)} labels)")
                st.rerun()  # Rerun to refresh the page and show updated data from all users
    
    #st.markdown("---")
    
    # Display current data
    st.header("Labeled Data")
    
    if len(df) > 0:
        # Format timestamp for display (remove microseconds)
        df_display = df.copy()
        if 'timestamp' in df_display.columns:
            # Convert timestamp to string format without microseconds
            def format_timestamp(ts):
                if pd.isna(ts):
                    return ts
                if isinstance(ts, str):
                    # If string, remove everything after the dot (microseconds)
                    if '.' in ts and 'T' in ts:
                        return ts.split('.')[0]
                    return ts
                elif isinstance(ts, datetime):
                    # If datetime object, format without microseconds
                    return ts.strftime("%Y-%m-%dT%H:%M:%S")
                return ts
            
            df_display['timestamp'] = df_display['timestamp'].apply(format_timestamp)
        
        # Show data table
        st.dataframe(df_display, width='stretch', hide_index=True)
        
        # Summary statistics
        with st.expander("📈 View Summary Statistics"):
            # Metrics    
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<h3 style='margin-bottom: 0;'>{len(df)}</h3>"
                    f"<p style='margin-top: 0; color: gray;'>Total Entries</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<h3 style='margin-bottom: 0;'>{df['toy'].nunique()}</h3>"
                    f"<p style='margin-top: 0; color: gray;'>Unique Toys</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<h3 style='margin-bottom: 0;'>{len(df[df['location_state'] != ''])}</h3>"
                    f"<p style='margin-top: 0; color: gray;'>With Location</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            if len(df) > 0:
                st.markdown("<h2 style='text-align: center;'>Toy Distribution</h2>", unsafe_allow_html=True)
                fig = plot_toy_frequency_analysis(df)
                if fig:
                    st.plotly_chart(fig, width='stretch')

            # PCA Analysis
            if len(df) >= 3:
                # Extract features
                df_features = extract_code_features(df)
                
                # Perform PCA with 3 components
                pca, pca_data, explained_variance, feature_names = perform_pca_analysis(df_features, n_components=3)
                
                if pca is not None and pca_data is not None:
                    # Initialize color_by in session state if not exists
                    if 'pca_color_by' not in st.session_state:
                        st.session_state.pca_color_by = 'TOY'
                    
                    # Number of clusters - initialize in session state if not exists
                    if 'pca_n_clusters' not in st.session_state:
                        st.session_state.pca_n_clusters = min(5, len(df) // 2) if len(df) > 0 else 5
                    
                    # 3D PCA Scatter Plot - centered title
                    st.markdown("<h2 style='text-align: center;'>3D PCA Visualization</h2>", unsafe_allow_html=True)
                    
                    # Color by selection - segmented control over full width (below title)
                    color_by = st.segmented_control(
                        "Color by:",
                        options=['TOY', 'CLUSTER'],
                        default=st.session_state.pca_color_by,
                        #help="Choose how to color points in the 3D plot",
                        width='stretch',
                        label_visibility="hidden"
                    )
                    st.session_state.pca_color_by = color_by
                    
                    # Number of clusters if needed
                    if color_by == 'CLUSTER':
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            n_clusters = st.number_input(
                                "Number of clusters:",
                                min_value=2,
                                max_value=min(10, len(df)),
                                value=st.session_state.pca_n_clusters,
                                step=1,
                                help="Number of clusters for KMeans clustering"
                            )
                            st.session_state.pca_n_clusters = n_clusters
                    else:
                        n_clusters = None
                    
                    # 3D PCA Scatter Plot - centered
                    col1, col2, col3 = st.columns([1, 10, 1])
                    with col2:
                        # Convert to lowercase for plot function
                        color_by_lower = st.session_state.pca_color_by.lower()
                        pca_fig = plot_pca_3d(pca_data, explained_variance, color_by=color_by_lower, n_clusters=n_clusters)
                        if pca_fig:
                            st.plotly_chart(pca_fig, width='stretch')
                    
                    # PCA Statistics - centered text
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f"<div style='text-align: center;'>"
                            f"<h3 style='margin-bottom: 0;'>{sum(explained_variance[:3]):.1%}</h3>"
                            f"<p style='margin-top: 0; color: gray;'>Total Variance Explained (PC1+PC2+PC3)</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"<div style='text-align: center;'>"
                            f"<h3 style='margin-bottom: 0;'>{len(feature_names)}</h3>"
                            f"<p style='margin-top: 0; color: gray;'>Number of Features</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f"<div style='text-align: center;'>"
                            f"<h3 style='margin-bottom: 0;'>{len(pca_data)}</h3>"
                            f"<p style='margin-top: 0; color: gray;'>Data Points</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Explained Variance
                    #st.subheader("Explained Variance by Component")
                    #variance_fig = plot_pca_variance_explained(explained_variance)
                    #if variance_fig:
                    #    st.plotly_chart(variance_fig, use_container_width=True)
                    
                    # PCA Loadings
                    st.markdown("<h2 style='text-align: center;'>PCA Feature Loadings</h2>", unsafe_allow_html=True)
                    st.caption("Shows how each feature contributes to the principal components")
                    loadings_fig = plot_pca_loadings(pca, feature_names, n_components=3)
                    if loadings_fig:
                        st.plotly_chart(loadings_fig, width='stretch')
                else:
                    st.warning("Not enough numerical features for PCA analysis. Need at least 2 features.")
            else:
                st.info("Need at least 3 data points for PCA analysis.")
    else:
        st.info("No labels yet. Start adding labels using the form above!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Stivanborčo Lab - "
        "<a href='https://cz.linkedin.com/in/václav-stibor-a26892293' target='_blank' style='color: gray; text-decoration: none;'>Václav Stibor</a>, "
        "<a href='https://cz.linkedin.com/in/filip-vančo-a675a4288' target='_blank' style='color: gray; text-decoration: none;'>Filip Vančo</a>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

