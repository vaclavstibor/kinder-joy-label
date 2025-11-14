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
5. The data is automatically saved to Firestore (if configured) or `labeled_data.csv` (fallback)
6. View all labeled data in the table below

## Data Storage

### ⚠️ Why CSV Storage Doesn't Work on Streamlit Cloud

Streamlit Cloud uses an **ephemeral filesystem** - all files (including CSV) are temporary and **will be deleted** when:
- The app restarts (happens automatically overnight or during maintenance)
- The app is redeployed
- The container is recycled

**This is why your data disappeared overnight!** CSV files are not persistent on Streamlit Cloud.

### Firebase Firestore (Recommended for Streamlit Cloud)

The app uses **Firebase Firestore** (NoSQL database) for persistent data storage, which prevents data loss when the app restarts or redeploys on Streamlit Cloud.

**Storage Limits (Firestore Free Tier):**
- ✅ **1 GB storage** (free tier)
- ✅ **50K reads/day** (free tier)
- ✅ **20K writes/day** (free tier)
- ✅ **20K deletes/day** (free tier)
- ✅ **Permanent storage** - data never disappears
- ✅ **Free** - generous free tier for most projects

For labeling data, this allows for **thousands of labels per day** - more than enough for most projects.

#### Setup Instructions:

1. **Create a Firebase Project**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Click "Add project" or select an existing project
   - Follow the setup wizard (you can disable Google Analytics if you don't need it)

2. **Create Firestore Database**:
   - In your Firebase project, go to "Firestore Database" in the left menu
   - Click "Create database"
   - Choose "Start in test mode" (for development) or "Start in production mode" (for production)
   - Select a location (choose the closest to your users)
   - Click "Enable"

3. **Create the Collection and Document** (Important!):
   
   **Struktura:**
   ```
   Collection: labels
     └── Document: records
         └── Field: labels (array of maps)
             └── [map1, map2, map3, ...]
                 Každý map obsahuje:
                 ├── timestamp (string)
                 ├── toy (string)
                 ├── balls_code (string)
                 ├── toy_code (string)
                 └── location_state (string)
   ```
   
   **Kroky:**
   - In the Firestore Database page, you'll see "Start collection" button
   - Click "Start collection"
   - **Collection ID**: Enter `labels` (exactly like this, lowercase)
   - Click "Next"
   - **Document ID**: Enter `records` (exactly like this, lowercase)
   - Click "Next"
   - **Add Field**:
     - **Field name**: `labels`
     - **Field type**: Select **"array"** (if available) or **"map"** (if array is not available)
     - **Value**: 
       - If **array**: Leave empty `[]` (the app will populate it automatically)
       - If **map**: You can add a test entry (see below)
   - Click "Save"
   
   **Test Entry (Optional, only if using "map" type):**
   If you want to add a test entry when using "map" type:
   - Click "Add field" inside the map
   - Field name: `0` (represents array index)
   - Field type: `map`
   - Inside this map, add these fields:
     - **Field 1**: `timestamp` (string): `"2024-01-01 12:00:00"`
     - **Field 2**: `toy` (string): `"Nancy"`
     - **Field 3**: `balls_code` (string): `"TEST001"`
     - **Field 4**: `toy_code` (string): `"TOY001"`
     - **Field 5**: `location_state` (string): `"Test"`
   
   **Note**: The collection and document will be created automatically when the app saves its first label. Creating it manually helps verify your setup.
   
   **Important**: All labels are stored in a **single document** (`records`) as an **array of maps** in the `labels` field. Each map in the array contains these fields (all strings):
   - `timestamp` - Date and time when label was created
   - `toy` - Selected toy name
   - `balls_code` - Balls code input
   - `toy_code` - Toy code input
   - `location_state` - Optional location state

4. **Create Service Account**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Select your Firebase project (it's the same as your Firebase project)
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name (e.g., "streamlit-firestore") and click "Create and Continue"
   - Grant role: "Cloud Datastore User" (or "Firestore User")
   - Click "Continue" then "Done"
   - Click on the created service account
   - Go to "Keys" tab > "Add Key" > "Create new key"
   - Choose JSON format and download the key file (save it as `firestore-key.json` for local development)

5. **Configure Streamlit Secrets**:
   - In your Streamlit Cloud app, go to "Settings" > "Secrets"
   - Add the following structure:
     ```toml
     [gcp_service_account]
     type = "service_account"
     project_id = "your-project-id"
     private_key_id = "your-private-key-id"
     private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
     client_email = "your-service-account@your-project.iam.gserviceaccount.com"
     client_id = "your-client-id"
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
     ```
   - Copy the values from your downloaded JSON key file

6. **Verify Collection Setup**:
   - The app uses collection name: `labels`
   - All labels are stored in a **single document** with ID: `records`
   - The document contains a field `labels` which is an **array of maps**
   - Each map in the array represents one label with fields: `timestamp`, `toy`, `balls_code`, `toy_code`, `location_state`
   - You can view all labels in Firebase Console > Firestore Database > `labels` collection > `records` document > `labels` array

### CSV Fallback

If Firestore is not configured, the app will automatically fall back to saving data in `labeled_data.csv`. However, **this data will be lost** when Streamlit Cloud restarts or redeploys the app.

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Configure Firestore secrets (see above) in "Settings" > "Secrets"
6. Click "Deploy"

The app will be available online and accessible to multiple users. **Data will persist in Firestore** even after app restarts.

### Viewing Your Data

You can view all labeled data in:
- **Firebase Console**: Go to Firestore Database > `labels` collection > `records` document > `labels` array to see all labels
- **Streamlit App**: The app displays all labels in a table below the form