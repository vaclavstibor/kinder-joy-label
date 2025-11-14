import streamlit as st
import pandas as pd
import os
from datetime import datetime
from google.cloud import firestore
from google.oauth2 import service_account
import json

# Page configuration
st.set_page_config(
    page_title="Kinder Joy Labeling App",
    page_icon="🎁",
    layout="wide"
)

# Toy list (24 toys)
TOYS = [
    "Nancy", "Mike", "Lucas", "Demogordon", "Steve", "Eleven", "Vecna", 
    "Eleven Down", "Max Down", "Eleven clip", "Demogordon pen", 
    "Steven and Robin pen", "Erica kabel", "Demogordon clip", "Will", "Max", 
    "Dustin", "Hopper", "Will Donw", "Steve Down", "Eddie Down", 
    "Dustin Down", "Hopper Down", "Robin Down"
]

# Firestore configuration
COLLECTION_NAME = "labels"  # Collection name in Firestore - each label is a separate document

def get_firestore_client():
    """Initialize Firestore client using Streamlit secrets or service account file"""
    try:
        # Check if credentials are in Streamlit secrets
        if 'gcp_service_account' in st.secrets:
            creds_info = st.secrets['gcp_service_account']
            # Convert to dict if it's not already
            if isinstance(creds_info, dict):
                project_id = creds_info.get('project_id', None)
                credentials = service_account.Credentials.from_service_account_info(creds_info)
            else:
                # If it's a string, try to parse it as JSON
                creds_dict = json.loads(creds_info) if isinstance(creds_info, str) else creds_info
                project_id = creds_dict.get('project_id', None)
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
            db = firestore.Client(credentials=credentials, project=project_id)
            return db
        elif os.path.exists("secrets.json"):
            # Try local file (for development)
            db = firestore.Client.from_service_account_json("secrets.json")
            return db
    except Exception as e:
        st.warning(f"Firestore not configured: {str(e)}")
        return None
    return None

@st.cache_data(ttl=5)  # Cache for 5 seconds - balances freshness with performance on Streamlit Cloud
def load_existing_data():
    """Load existing labeled data from Firestore (each label is a separate document)"""
    db = get_firestore_client()
    if not db:
        st.error("Firestore is not configured. Please set up your credentials in Streamlit Secrets.")
        return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])
    
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
            required_columns = ["timestamp", "toy", "balls_code", "toy_code", "location_state"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Filter out empty rows
            df = df.dropna(how='all')
            df = df[~(df.astype(str).apply(lambda x: x.str.strip().eq('')).all(axis=1))]
            return df[required_columns]  # Return only required columns in correct order
        else:
            return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])
    except Exception as e:
        st.error(f"Error loading from Firestore: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])

def save_backup(new_row):
    """Save individual submitted item to Firestore (as a separate document)"""
    db = get_firestore_client()
    if not db:
        st.error("Firestore is not configured. Please set up your credentials in Streamlit Secrets.")
        return False
    
    try:
        # Check for duplicates before adding
        # Query for existing document with same data
        existing_docs = db.collection(COLLECTION_NAME)\
            .where('timestamp', '==', new_row['timestamp'])\
            .where('balls_code', '==', new_row['balls_code'])\
            .where('toy_code', '==', new_row['toy_code'])\
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
        "<h1 style='text-align: center; margin-bottom: 2rem;'>Stranger Things Labeling Dataset</h1>",
        unsafe_allow_html=True
    )

    # Description
    st.markdown(
        "<p style='text-align: center; margin-bottom: 2rem;'>This app is used to make labeled Stranger Things Kinder Joy toys dataset which will be used to make statistical predictive model.</p>",
        unsafe_allow_html=True
    )

    
    # Load existing data
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
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
        
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
                
                # Reload latest data to get any concurrent submissions from other users
                load_existing_data.clear()  # Clear cache first to get fresh data
                
                # Save new row to Firestore
                save_backup(new_row)
                
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
        # Show data table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary statistics
        with st.expander("📈 View Summary Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entries", len(df))
            with col2:
                st.metric("Unique Toys", df["toy"].nunique())
            with col3:
                st.metric("With Location", len(df[df["location_state"] != ""]))
            
            if len(df) > 0:
                st.subheader("Toy Distribution")
                toy_counts = df["toy"].value_counts()
                st.bar_chart(toy_counts)
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

