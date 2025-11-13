# Kinder Joy Labeling App

Streamlit application for labeling Kinder Joy toys with features.

## Features

- **24 Toys**: Nancy, Mike, Lucas, Demogordon, Steve, Eleven, Vecna, Eleven Down, Max Down, Eleven clip, Demogorgon pen, Steven a Robin pen, Eica kabel, Demogorgon clip, Will, Max, Dustin, Hopper, Will Donw, Steve Down, Eddie Down, Dustin Down, Hopper Down, Robin Down
- **Required Fields**:
  - `balls_code` (string) - Must be at least 1 character
  - `toy_code` (string) - Must be at least 1 character
- **Optional Field**:
  - `location_state` (string)

## How to Start the Application

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser**:
   The app will automatically open in your default browser at `http://localhost:8501`

### Alternative: Run with specific port
```bash
streamlit run streamlit_app.py --server.port 8501
```

## Usage

1. Select a toy from the dropdown menu
2. Enter the required `balls_code` and `toy_code`
3. Optionally enter `location_state`
4. Click "Submit Label" or press Enter
5. The data is automatically saved to `labeled_data.csv`
6. Download the CSV dataset anytime using the download button

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Click "Deploy"

The app will be available online and accessible to multiple users.