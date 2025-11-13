import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Kinder Joy Labeling App",
    page_icon="🎁",
    layout="wide"
)

# Toy list (24 toys)
TOYS = [
    "Nancy", "Mike", "Lucas", "Demogordon", "Steve", "Eleven", "Vecna", 
    "Eleven Down", "Max Down", "Eleven clip", "Demogorgon pen", 
    "Steven a Robin pen", "Eica kabel", "Demogorgon clip", "Will", "Max", 
    "Dustin", "Hopper", "Will Donw", "Steve Down", "Eddie Down", 
    "Dustin Down", "Hopper Down", "Robin Down"
]

# CSV file path
CSV_FILE = "labeled_data.csv"

def load_existing_data():
    """Load existing labeled data from CSV"""
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except:
            return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])
    return pd.DataFrame(columns=["timestamp", "toy", "balls_code", "toy_code", "location_state"])

def save_data(df):
    """Save DataFrame to CSV"""
    df.to_csv(CSV_FILE, index=False)

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
                
                # Add to DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save to CSV
                save_data(df)
                
                st.success(f"✅ Label saved successfully! (Total: {len(df)} labels)")
    
    st.markdown("---")
    
    # Display current data
    st.header("Labeled Data")
    
    if len(df) > 0:
        # Show data table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        # csv = df.to_csv(index=False).encode('utf-8')
        # st.download_button(
        #     label="📥 Download CSV",
        #     data=csv,
        #     file_name=f"labeled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        #     mime="text/csv",
        #     use_container_width=True
        # )
        
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

