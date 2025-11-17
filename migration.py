"""
Migration script for Firestore data structure
Moves individual label documents to a single aggregated document

This script can be run manually if needed in the future.
"""

from google.cloud import firestore
from google.oauth2 import service_account
from datetime import datetime
import streamlit as st

# Configuration
COLLECTION_NAME = "labels"  # Source collection with individual documents
SINGLE_DOC_COLLECTION = "all_labels_in_one"  # Target collection for aggregated document
SINGLE_DOC_ID = "all_labels"  # Document ID for aggregated data


def migrate_to_single_document(creds_dict=None):
    """
    Manually migrate all individual documents to single document format
    
    Args:
        creds_dict: Optional service account credentials dict
                    If None, will try to use Streamlit secrets
    """
    print("\n" + "="*70)
    print("MIGRATION STARTED")
    print("="*70)
    
    # Initialize Firestore client
    if creds_dict:
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        db = firestore.Client(credentials=credentials, project=creds_dict.get('project_id'))
    else:
        # Try to use Streamlit secrets if available
        try:
            if 'gcp_service_account' in st.secrets:
                creds_dict = dict(st.secrets['gcp_service_account'])
                if 'private_key' in creds_dict and isinstance(creds_dict['private_key'], str):
                    creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
                credentials = service_account.Credentials.from_service_account_info(creds_dict)
                db = firestore.Client(credentials=credentials, project=creds_dict.get('project_id'))
            else:
                print("❌ ERROR: No credentials provided and Streamlit secrets not available.")
                return False
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize Firestore client: {e}")
            return False
    
    try:
        doc_ref = db.collection(SINGLE_DOC_COLLECTION).document(SINGLE_DOC_ID)
        
        # Check if single document already exists
        print(f"📋 Checking for existing single document: {SINGLE_DOC_ID}")
        existing_doc = doc_ref.get()
        existing_labels = []
        if existing_doc.exists:
            data = existing_doc.to_dict()
            existing_labels = data.get('labels', [])
            if existing_labels:
                print(f"⚠️  WARNING: Single document already exists with {len(existing_labels)} labels. Will merge and remove duplicates.")
        
        # Read all individual documents
        print(f"🔄 Reading all individual documents from collection '{COLLECTION_NAME}'...")
        docs = db.collection(COLLECTION_NAME).stream()
        records = []
        doc_count = 0
        
        for doc in docs:
            # Skip any document that might be the single document (shouldn't be in labels collection)
            doc_data = doc.to_dict()
            if doc_data:
                records.append(doc_data)
                doc_count += 1
                if doc_count % 10 == 0:
                    print(f"   📄 Processed {doc_count} documents...")
        
        print(f"✅ Found {doc_count} individual documents to migrate")
        
        if records:
            # Merge with existing labels if any
            if existing_doc.exists and existing_labels:
                print(f"📊 Merging with existing {len(existing_labels)} labels...")
                # Combine and remove duplicates
                all_records = existing_labels + records
                # Remove duplicates based on timestamp, balls_code, toy_code
                seen = set()
                unique_records = []
                for record in all_records:
                    key = (record.get('timestamp'), record.get('balls_code'), record.get('toy_code'))
                    if key not in seen:
                        seen.add(key)
                        unique_records.append(record)
                records = unique_records
                print(f"📊 After deduplication: {len(records)} unique labels (was {len(existing_labels)} existing + {doc_count} new)")
            
            # Write to single document
            print(f"💾 Writing {len(records)} labels to single document '{SINGLE_DOC_ID}'...")
            doc_ref.set({
                'labels': records,
                'migrated_at': datetime.now().isoformat(),
                'total_labels': len(records)
            })
            
            print("="*70)
            print(f"✅ MIGRATION COMPLETE!")
            print(f"   • Migrated {doc_count} individual documents")
            print(f"   • Total labels in single document: {len(records)}")
            print(f"   • Document path: {SINGLE_DOC_COLLECTION}/{SINGLE_DOC_ID}")
            print("="*70)
            print("💡 You can now delete the old individual documents from Firestore console to clean up.\n")
            
            return True
        else:
            print("⚠️  WARNING: No individual documents found to migrate.")
            return False
            
    except Exception as e:
        print("="*70)
        print(f"❌ MIGRATION FAILED: {e}")
        print("="*70)
        import traceback
        print(traceback.format_exc())
        print("="*70 + "\n")
        return False


if __name__ == "__main__":
    # Can be run standalone if credentials are provided via environment or file
    print("Migration script - use migrate_to_single_document() function")
    print("Or run from Streamlit app Admin Tools section")


"""
# Admin section
with st.expander("🔧 Admin Tools", expanded=False):
    tab1, tab2 = st.tabs(["Migration", "Firestore Diagnostics"])
    
    with tab1:
        st.markdown("**Optimize Firestore reads:** Migrate all individual documents to a single document format.")
        st.markdown("This will reduce reads from N documents to just 1 read per page load.")
        st.markdown("⚠️ **Note:** After migration, you can delete the old individual documents from Firestore console.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Run Migration", type="primary", use_container_width=True):
                with st.spinner("Running migration..."):
                    if migrate_to_single_document():
                        st.rerun()
        with col2:
            if st.button("🔄 Clear Cache & Reload", use_container_width=True):
                load_existing_data.clear()
                st.success("Cache cleared! Reloading data...")
                st.rerun()
    
    with tab2:
        st.markdown("**Firestore Connection Diagnostics**")
        db = get_firestore_client()
        
        if db:
            try:
                # Get project ID
                project_id = db.project
                st.success(f"✅ Connected to Firestore project: `{project_id}`")
                
                st.markdown(f"**Labels collection:** `{COLLECTION_NAME}`")
                st.markdown(f"**Single document collection:** `{SINGLE_DOC_COLLECTION}`")
                st.markdown(f"**Single document ID:** `{SINGLE_DOC_ID}`")
                st.markdown(f"**Full document path:** `{SINGLE_DOC_COLLECTION}/{SINGLE_DOC_ID}`")
                
                # Check if collection exists by trying to list documents
                st.markdown("---")
                st.markdown("**Checking collection...**")
                
                try:
                    # Try to get all documents (limited for performance)
                    all_docs = list(db.collection(COLLECTION_NAME).stream())
                    doc_count = len(all_docs)
                    sample_docs = all_docs[:5]  # Show first 5
                    
                    st.info(f"📊 Found **{doc_count}** document(s) in collection `{COLLECTION_NAME}`")
                    
                    if doc_count > 0:
                        st.markdown("**Sample documents (first 5):**")
                        for i, doc in enumerate(sample_docs, 1):
                            doc_data = doc.to_dict()
                            st.code(f"Document ID: {doc.id}\nData: {doc_data}")
                    
                    # Check for single document
                    single_doc_ref = db.collection(SINGLE_DOC_COLLECTION).document(SINGLE_DOC_ID)
                    single_doc = single_doc_ref.get()
                    
                    if single_doc.exists:
                        single_data = single_doc.to_dict()
                        label_count = len(single_data.get('labels', []))
                        st.success(f"✅ Single document `{SINGLE_DOC_ID}` exists with {label_count} labels")
                    else:
                        st.warning(f"⚠️ Single document `{SINGLE_DOC_ID}` does not exist yet")
                        st.info("💡 Run migration to create it")
                
                except Exception as e:
                    st.error(f"❌ Error accessing collection: {e}")
                    st.info("💡 Make sure:")
                    st.markdown("1. The collection name is correct")
                    st.markdown("2. Your service account has Firestore read/write permissions")
                    st.markdown("3. You're looking at the correct GCP project in Firestore console")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.error("❌ Firestore client not initialized")
            st.info("Check your secrets.toml configuration")

# Migration function moved to migration.py for future use
# Uncomment and import if needed:
# from migration import migrate_to_single_document

def migrate_to_single_document():
    #Manually migrate all individual documents to single document format - moved to migration.py
    # Import migration function from separate file
    try:
        from migration import migrate_to_single_document as migrate_func
        # Get credentials from Streamlit secrets
        if 'gcp_service_account' in st.secrets:
            creds_dict = dict(st.secrets['gcp_service_account'])
            if 'private_key' in creds_dict and isinstance(creds_dict['private_key'], str):
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            
            result = migrate_func(creds_dict=creds_dict)
            if result:
                st.success("✅ Migration complete!")
                load_existing_data.clear()
                st.rerun()
            else:
                st.warning("⚠️ Migration completed with warnings or no data to migrate.")
        else:
            st.error("❌ Firestore credentials not found in secrets.")
    except ImportError:
        st.error("❌ Migration module not found. Please ensure migration.py exists.")
    except Exception as e:
        st.error(f"❌ Migration failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
"""