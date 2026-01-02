import streamlit as st
import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime
from io import StringIO
import sys
import os
from typing import Dict, List
from dotenv import load_dotenv
import requests

# Import normalization functions from normalize.py
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv()

from normalize import _load_json
from revyops_integration import push_companies_to_revyops, verify_companies_in_revyops

MAPPINGS_DIR = ROOT / "mappings"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_JSON = MAPPINGS_DIR / "canonical_columns.json"
MAPPINGS_CSV = MAPPINGS_DIR / "mapping.csv"

# Source configuration
SOURCE_CONFIG = {
    "crunchbase": {"label": "Crunchbase", "required": False},
    "discolikes": {"label": "Discolikes", "required": False},
    "aiark": {"label": "AIARK", "required": False},
    "apollo": {"label": "Apollo", "required": False},
    "linkedin": {"label": "LinkedIn", "required": False},
    "googlemaps": {"label": "Google Maps", "required": False}
}

SOURCE_VALUES = {
    "crunchbase": "Crunchbase",
    "discolikes": "Discolikes",
    "aiark": "AI-Ark",
    "apollo": "Apollo",
    "linkedin": "LinkedIn",
    "googlemaps": "Google Maps"
}


def _load_mappings_from_csv() -> Dict[str, Dict[str, str]]:
    """Load mappings from CSV file"""
    mappings_df = pd.read_csv(MAPPINGS_CSV)
    canonical_col = "RevyOps Schema (Map Headers Here)"
    source_columns = {
        "crunchbase": "Crunchbase Headers",
        "discolikes": "Discolikes Headers",
        "aiark": "AIARK Headers",
        "apollo": "Apollo Headers",
        "linkedin": "LinkedIn Headers",
        "googlemaps": "Google Maps Headers"
    }
    
    result = {}
    for source_name, source_col in source_columns.items():
        mapping = {}
        for _, row in mappings_df.iterrows():
            canonical = str(row[canonical_col]).strip()
            source_header = str(row[source_col]).strip()
            if pd.isna(row[source_col]) or source_header == "" or source_header == "nan":
                continue
            mapping[source_header] = canonical
        result[source_name] = mapping
    return result


def _process_source_df(df: pd.DataFrame, source_name: str, mapping: Dict[str, str], canonical: List[str]) -> pd.DataFrame:
    """Process a single source dataframe"""
    from normalize import (
        _apply_mapping, _ensure_canonical_columns, _normalize_domain,
        _extract_linkedin_url, _extract_facebook_url, _extract_twitter_url,
        _domain_to_website_url
    )
    
    # Apply mapping
    df = _apply_mapping(df.copy(), mapping)
    
    # Source-specific processing
    if source_name == "crunchbase":
        if "Website URL" in df.columns:
            df["Domain"] = df["Website URL"].map(_normalize_domain)
    elif source_name == "discolikes":
        if "All Social URL's" in df.columns:
            df["LinkedIn URL"] = df["All Social URL's"].map(_extract_linkedin_url)
            df["Facebook URL"] = df["All Social URL's"].map(_extract_facebook_url)
            df["Twitter URL"] = df["All Social URL's"].map(_extract_twitter_url)
        if "Domain" in df.columns:
            df["Website URL"] = df["Domain"].map(_domain_to_website_url)
    elif source_name == "aiark":
        if "Domain" in df.columns:
            df["Website URL"] = df["Domain"].map(_domain_to_website_url)
        elif "Website URL" in df.columns:
            df["Domain"] = df["Website URL"].map(_normalize_domain)
    elif source_name == "apollo":
        # Map "Website" to "Website URL" (handled by mapping)
        # Extract domain from Website URL
        if "Website URL" in df.columns:
            # Extract domain from Website URL, leave empty if Website URL is empty
            df["Domain"] = df["Website URL"].apply(
                lambda x: _normalize_domain(x) if pd.notna(x) and str(x).strip() != "" else ""
            )
    elif source_name == "linkedin":
        if "Domain" in df.columns:
            df["Website URL"] = df["Domain"].map(_domain_to_website_url)
    elif source_name == "googlemaps":
        # After mapping, "website" is mapped to "Domain" column
        # Extract domain from URL values in Domain column
        if "Domain" in df.columns:
            # Store original URL as Website URL before normalizing
            df["Website URL"] = df["Domain"]
            # Normalize Domain to extract just the domain name
            df["Domain"] = df["Domain"].map(_normalize_domain)
    
    # Ensure canonical columns
    df = _ensure_canonical_columns(df, canonical)
    
    # Normalize domains
    if "Domain" in df.columns:
        df["Domain"] = df["Domain"].map(_normalize_domain)
    
    # Add Source column
    df["Source"] = SOURCE_VALUES.get(source_name, source_name)
    
    return df


def normalize_dataframes(source_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Normalize and merge multiple source dataframes"""
    from normalize import _dedupe_keep_most_complete
    
    canonical = _load_json(CANONICAL_JSON)
    all_mappings = _load_mappings_from_csv()
    
    processed_dfs = []
    
    for source_name, df in source_dataframes.items():
        if source_name not in all_mappings:
            continue
        mapping = all_mappings[source_name]
        processed_df = _process_source_df(df, source_name, mapping, canonical)
        processed_dfs.append(processed_df)
    
    if not processed_dfs:
        raise ValueError("No valid source files to process!")
    
    # Merge and dedupe
    merged = pd.concat(processed_dfs, ignore_index=True)
    merged = _dedupe_keep_most_complete(merged)
    
    return merged


def clean_company_name(name: str) -> str:
    """
    Clean and normalize company names using the specified formula.
    Steps:
    1. Split by comma and take first part
    2. Remove common company suffixes
    3. Remove everything after pipe
    4. Remove patterns like " - word word" at end
    5. Remove domain extensions
    6. Remove special characters (keep word chars, spaces, &, -, ', +)
    7. Replace multiple spaces with single space
    8. Remove trailing spaces, dashes, and ampersands
    9. Trim
    """
    if not name or pd.isna(name):
        return ""
    
    name = str(name).strip()
    if not name:
        return ""
    
    # Step 1: Split by comma and take first part
    name = name.split(",")[0] if "," in name else name
    name = name.strip()
    if not name:
        return ""
    
    # Step 2: Remove common company suffixes (case-insensitive, at end of string)
    suffixes = [
        r'\s+inc\.?\s*$', r'\s+incorporated\s*$', r'\s+corp\.?\s*$', r'\s+corporation\s*$',
        r'\s+llc\s*$', r'\s+l\.l\.c\.?\s*$', r'\s+plc\s*$', r'\s+limited\s*$', r'\s+ltd\.?\s*$',
        r'\s+gmbh\s*$', r'\s+s\.a\.s\.?\s*$', r'\s+s\.a\.?\s*$', r'\s+s\.r\.l\.?\s*$', r'\s+sarl\s*$',
        r'\s+s\.r\.o\.?\s*$', r'\s+bv\s*$', r'\s+bvba\s*$', r'\s+nv\s*$', r'\s+ag\s*$', r'\s+oy\s*$',
        r'\s+ab\s*$', r'\s+as\s*$', r'\s+psc\s*$', r'\s+pte\.?\s*ltd\.?\s*$', r'\s+pty\.?\s*ltd\.?\s*$',
        r'\s+oyj\s*$', r'\s+kk\s*$', r'\s+k\.k\.?\s*$', r'\s+llp\s*$', r'\s+lp\s*$', r'\s+holdings?\s*$',
        r'\s+spa\s*$', r'\s+s\.p\.a\.?\s*$'
    ]
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    name = name.strip()
    
    # Additional LLC removal - catch any remaining LLC variations (more aggressive)
    name = re.sub(r'\s+llc\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+l\.l\.c\.?\s*$', '', name, flags=re.IGNORECASE)
    name = name.strip()
    
    # Step 3: Remove everything after pipe
    name = re.sub(r'\s*\|.*$', '', name)
    name = name.strip()
    
    # Step 4: Remove patterns like " - word word" and everything after at end
    name = re.sub(r'\s+-\s+\w+\s+\w+.*$', '', name)
    name = name.strip()
    
    # Step 5: Remove domain extensions at end
    name = re.sub(r'\.(com|net|org|io|co|ai|app|dev|tech|biz|info)\s*$', '', name, flags=re.IGNORECASE)
    name = name.strip()
    
    # Step 6: Remove special characters (keep word chars, spaces, &, -, ', +)
    # Keep: word characters (\w), whitespace (\s), &, -, ', +
    name = re.sub(r'[^\w\s&\-\'\+]', '', name)
    
    # Step 7: Replace multiple spaces with single space
    name = re.sub(r'\s{2,}', ' ', name)
    
    # Step 8: Remove trailing spaces, dashes, and ampersands
    name = re.sub(r'[\s\-&]+$', '', name)
    
    # Step 9: Final LLC cleanup - ensure LLC is removed even if it survived previous steps
    # Try multiple patterns to catch all variations
    name = re.sub(r'\s+llc\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+l\.l\.c\.?\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+LLC\s*$', '', name)  # Explicit uppercase
    name = re.sub(r'\s+L\.L\.C\.?\s*$', '', name)  # Explicit uppercase with dots
    # Also handle cases where LLC might be directly attached (edge case)
    name = re.sub(r'LLC\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'L\.L\.C\.?\s*$', '', name, flags=re.IGNORECASE)
    
    # Step 10: Trim and return
    name = name.strip()
    
    return name


def read_csv_from_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV from uploaded file"""
    try:
        # Try UTF-8 first
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1
        uploaded_file.seek(0)  # Reset file pointer
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, encoding="latin-1")


# Password Authentication
# Get password from Streamlit secrets (for cloud deployment) or environment variable, or use default
try:
    APP_PASSWORD = st.secrets.get("app_password", os.getenv("APP_PASSWORD", "Buzzlead23$"))
except:
    # Fallback if secrets not available (local development)
    APP_PASSWORD = os.getenv("APP_PASSWORD", "Buzzlead23$")

# Check authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Company CSV DropBox - Login", page_icon="🔒", layout="centered")
    st.title("🔒 Company CSV DropBox")
    st.markdown("Please enter the password to access the application.")
    
    password_input = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login", type="primary", use_container_width=True):
        if password_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        elif password_input:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Company CSV DropBox", page_icon="📊", layout="wide")

st.title("📊 Company CSV DropBox")
st.markdown("Upload CSV files from multiple sources to normalize and merge them.")

# File upload section - 6 sources in 2 columns
col1, col2 = st.columns(2)

uploaded_files = {}

with col1:
    for source_name in ["crunchbase", "discolikes", "aiark"]:
        config = SOURCE_CONFIG[source_name]
        uploaded_files[source_name] = st.file_uploader(
            f"📁 {config['label']} CSV",
            type=['csv'],
            key=source_name,
            accept_multiple_files=True,
            help=f"Upload one or more {config['label']} export files (optional). You can upload multiple files for the same source."
        )

with col2:
    for source_name in ["apollo", "linkedin", "googlemaps"]:
        config = SOURCE_CONFIG[source_name]
        uploaded_files[source_name] = st.file_uploader(
            f"📁 {config['label']} CSV",
            type=['csv'],
            key=source_name,
            accept_multiple_files=True,
            help=f"Upload one or more {config['label']} export files (optional). You can upload multiple files for the same source."
        )

# Client Name input
st.subheader("👤 Client Settings")
client_name = st.text_input(
    "Client Name",
    value="",
    help="Enter the client name (e.g., 'Buzzlead'). A folder will be created with this name if it doesn't exist.",
    placeholder="e.g., Buzzlead"
)

# Vertical and Date inputs
st.subheader("✏️ Output Settings")
vertical = st.text_input(
    "Vertical",
    value="",
    help="Enter the vertical name (e.g., 'Screen Print Shops' or 'PR Firms').",
    placeholder="e.g., Screen Print Shops"
)

date = st.text_input(
    "Date",
    value="",
    help="Enter the date (e.g., '12-23-25').",
    placeholder="e.g., 12-23-25"
)

# RevyOps API Key input
st.subheader("🔗 RevyOps Integration")

# Client name to workspace mapping (with aliases)
CLIENT_WORKSPACE_MAPPING = {
    "Buzzlead": "Buzzlead",
    "Product EVO": "Product EVO",
    "EVO": "Product EVO",
    "Incentive Solutions Corp": "Incentive Solutions Corp",
    "Incentive Solutions": "Incentive Solutions Corp",
    "Interdependence": "Interdependence",
    "IDPR": "Interdependence",
}

# Default workspace for clients not in the mapping
DEFAULT_WORKSPACE = "Master Account (Excludes IDPR, EVO, Incentives, Buzzlead)"

# Load API keys from Streamlit Secrets (for cloud deployment) or config file (for local development)
workspace_api_keys = {}
try:
    # Try to load from Streamlit Secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'revyops_keys' in st.secrets:
        workspace_api_keys = dict(st.secrets['revyops_keys'])
except:
    pass

# Fallback to local file if secrets not available
if not workspace_api_keys:
    revyops_keys_file = ROOT / "revyops_keys.json"
    if revyops_keys_file.exists():
        try:
            with open(revyops_keys_file, 'r') as f:
                workspace_api_keys = json.load(f)
        except:
            workspace_api_keys = {}

# Determine which workspace to use based on client name
client_name_normalized = client_name.strip()
workspace_name = CLIENT_WORKSPACE_MAPPING.get(client_name_normalized, DEFAULT_WORKSPACE)

# Get API key for the workspace (check Streamlit Secrets first, then local file, then env var)
default_api_key = ""
if workspace_name in workspace_api_keys:
    default_api_key = workspace_api_keys[workspace_name]
elif os.getenv("REVYOPS_API_KEY"):
    default_api_key = os.getenv("REVYOPS_API_KEY")

# Show which workspace will be used
if client_name.strip():
    st.caption(f"📍 Workspace: **{workspace_name}**")

# Use session state to track if we should update the API key field
# When client name changes, update the API key
if 'last_client_name' not in st.session_state:
    st.session_state['last_client_name'] = ""
    st.session_state['revyops_api_key'] = default_api_key

# If client name changed, update the API key
if client_name_normalized != st.session_state.get('last_client_name', ''):
    st.session_state['last_client_name'] = client_name_normalized
    st.session_state['revyops_api_key'] = default_api_key

revyops_api_key = st.text_input(
    "RevyOps API Key",
    value=st.session_state.get('revyops_api_key', default_api_key),
    type="password",
    key=f"revyops_key_{workspace_name}",  # Dynamic key forces update when workspace changes
    help=f"API key for {workspace_name}. Auto-loaded from Streamlit Secrets (cloud) or revyops_keys.json (local). You can also set a default in .env as REVYOPS_API_KEY.",
    placeholder="Enter API key or it will auto-load from config"
)

# Update session state when user manually changes the key
if revyops_api_key != st.session_state.get('revyops_api_key', ''):
    st.session_state['revyops_api_key'] = revyops_api_key

# Store API key in session state for use in push section
if 'revyops_api_key' not in st.session_state:
    st.session_state['revyops_api_key'] = revyops_api_key

# Preview uploaded files
uploaded_count = 0
for source_name, uploaded_file_list in uploaded_files.items():
    if uploaded_file_list is not None and len(uploaded_file_list) > 0:
        config = SOURCE_CONFIG[source_name]
        # Handle both single file (list with one item) and multiple files
        file_list = uploaded_file_list if isinstance(uploaded_file_list, list) else [uploaded_file_list]
        uploaded_count += len(file_list)
        
        if len(file_list) == 1:
            st.success(f"✅ {config['label']} file uploaded: {file_list[0].name}")
        else:
            st.success(f"✅ {config['label']}: {len(file_list)} files uploaded")
            for i, file in enumerate(file_list, 1):
                st.caption(f"  {i}. {file.name}")
        
        # Show preview for first file
        try:
            preview_df = read_csv_from_upload(file_list[0])
            total_rows = len(preview_df)
            # If multiple files, try to estimate total rows (just show first file's count)
            if len(file_list) > 1:
                st.info(f"📊 {config['label']} preview (first file): {total_rows} rows, {len(preview_df.columns)} columns ({len(file_list)} files total)")
            else:
                st.info(f"📊 {config['label']} preview: {total_rows} rows, {len(preview_df.columns)} columns")
            file_list[0].seek(0)  # Reset for processing
        except Exception as e:
            st.error(f"Error reading {config['label']} file: {str(e)}")

if uploaded_count == 0:
    st.warning("⚠️ No files uploaded yet. Please upload at least one CSV file.")

# Process button
if st.button("🚀 Normalize & Merge Files", type="primary", use_container_width=True):
    if uploaded_count == 0:
        st.error("❌ Please upload at least one CSV file before processing.")
    elif not client_name.strip():
        st.error("❌ Please enter a client name before processing.")
    elif not vertical.strip():
        st.error("❌ Please enter a vertical before processing.")
    elif not date.strip():
        st.error("❌ Please enter a date before processing.")
    else:
        with st.spinner("🔄 Processing files..."):
            try:
                # Sanitize client name for folder name (remove invalid characters)
                safe_client_name = re.sub(r'[<>:"/\\|?*]', '_', client_name.strip())
                if not safe_client_name:
                    st.error("❌ Invalid client name. Please use alphanumeric characters.")
                    st.stop()
                
                # Sanitize vertical and date for folder names
                safe_vertical = re.sub(r'[<>:"/\\|?*]', '_', vertical.strip())
                safe_date = re.sub(r'[<>:"/\\|?*]', '_', date.strip())
                
                # Create client folder structure
                client_dir = ROOT / safe_client_name
                client_dir.mkdir(parents=True, exist_ok=True)
                
                # Determine where to save input files
                # Check if Inputs folder exists - if not, this is the first run
                inputs_dir = client_dir / "Inputs"
                if not inputs_dir.exists():
                    # First run - use Inputs/ folder
                    input_files_dir = inputs_dir
                    input_files_dir.mkdir(parents=True, exist_ok=True)
                    is_first_run = True
                else:
                    # Subsequent run - create Draft subfolder
                    draft_folder_name = f"{safe_client_name} {safe_vertical} {safe_date} Draft"
                    input_files_dir = client_dir / draft_folder_name
                    input_files_dir.mkdir(parents=True, exist_ok=True)
                    is_first_run = False
                
                # Read all uploaded files and save them to Inputs folder
                source_dataframes = {}
                saved_files = []
                for source_name, uploaded_file_list in uploaded_files.items():
                    if uploaded_file_list is not None and len(uploaded_file_list) > 0:
                        # Handle both single file (list with one item) and multiple files
                        file_list = uploaded_file_list if isinstance(uploaded_file_list, list) else [uploaded_file_list]
                        
                        # Read and combine all files for this source
                        dfs_for_source = []
                        for uploaded_file in file_list:
                            try:
                                # Read the dataframe
                                df = read_csv_from_upload(uploaded_file)
                                # Remove "Tagline" and "Source" columns from input files
                                if "Tagline" in df.columns:
                                    df = df.drop(columns=["Tagline"])
                                if "Source" in df.columns:
                                    df = df.drop(columns=["Source"])
                                dfs_for_source.append(df)
                                
                                # Save uploaded file to appropriate folder
                                # Use original filename (no source prefix needed since files are in organized folders)
                                input_filename = uploaded_file.name
                                input_path = input_files_dir / input_filename
                                
                                # Handle duplicate filenames by appending a number
                                counter = 1
                                original_path = input_path
                                while input_path.exists():
                                    name_parts = original_path.stem, original_path.suffix
                                    input_filename = f"{name_parts[0]}_{counter}{name_parts[1]}"
                                    input_path = input_files_dir / input_filename
                                    counter += 1
                                
                                uploaded_file.seek(0)  # Reset file pointer
                                with open(input_path, 'wb') as f:
                                    f.write(uploaded_file.getbuffer())
                                saved_files.append(input_filename)
                            except Exception as e:
                                st.warning(f"⚠️ Error processing {uploaded_file.name} for {source_name}: {str(e)}")
                                continue
                        
                        # Combine all dataframes for this source
                        if dfs_for_source:
                            if len(dfs_for_source) == 1:
                                source_dataframes[source_name] = dfs_for_source[0]
                            else:
                                # Concatenate multiple files for the same source
                                import pandas as pd
                                combined_df = pd.concat(dfs_for_source, ignore_index=True)
                                source_dataframes[source_name] = combined_df
                                st.info(f"📊 Combined {len(dfs_for_source)} {SOURCE_CONFIG[source_name]['label']} files: {len(combined_df)} total rows")
                
                # Normalize
                normalized_df = normalize_dataframes(source_dataframes)
                
                # Clean and normalize company names
                if "Name" in normalized_df.columns:
                    normalized_df["Name"] = normalized_df["Name"].apply(clean_company_name)
                
                # Determine output filename: {Client Name} {Vertical} Accounts {Date}.csv
                output_filename = f"{client_name.strip()} {vertical.strip()} Accounts {date.strip()}.csv"
                tagline_value = f"{client_name.strip()} {vertical.strip()} Accounts {date.strip()}"
                
                # Set Tagline column to the output filename value (without .csv extension)
                if "Tagline" in normalized_df.columns:
                    normalized_df["Tagline"] = tagline_value
                else:
                    # If Tagline column doesn't exist, add it
                    normalized_df["Tagline"] = tagline_value
                
                # Set Client column to the client name value
                if "Client" in normalized_df.columns:
                    normalized_df["Client"] = client_name.strip()
                else:
                    # If Client column doesn't exist, add it
                    normalized_df["Client"] = client_name.strip()
                
                # Save output to client folder
                output_path = client_dir / output_filename
                normalized_df.to_csv(output_path, index=False, encoding="utf-8")
                
                # Store normalized_df in session state for RevyOps push
                st.session_state['normalized_df'] = normalized_df
                st.session_state['output_filename'] = output_filename
                st.session_state['processing_complete'] = True
                
                st.success(f"✅ Success! Processed {len(normalized_df)} rows")
                st.markdown(f"### 📄 Output File: `{output_filename}`")
                st.info(f"📁 **Full Path:** `{safe_client_name}/{output_filename}`")
                if saved_files:
                    if is_first_run:
                        st.info(f"📂 Input files saved to: `{safe_client_name}/Inputs/`")
                    else:
                        draft_folder_name = f"{safe_client_name} {safe_vertical} {safe_date} Draft"
                        st.info(f"📂 Input files saved to: `{safe_client_name}/{draft_folder_name}/`")
                    st.caption(f"Saved files: {', '.join(saved_files)}")
                
                # Display preview
                st.subheader("📋 Preview of Normalized Data")
                st.dataframe(normalized_df.head(10), use_container_width=True)
                
                # Download button
                csv_string = normalized_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Normalized CSV",
                    data=csv_string,
                    file_name=output_filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.exception(e)

# Show RevyOps Push section if processing is complete (persists across reruns)
if st.session_state.get('processing_complete') and 'normalized_df' in st.session_state:
    normalized_df = st.session_state['normalized_df']
    output_filename = st.session_state.get('output_filename', 'output.csv')
    
    # RevyOps Push button
    st.divider()
    st.subheader("🚀 Push to RevyOps")
    
    # Preview RevyOps mapping
    with st.expander("🔍 Preview RevyOps Mapping", expanded=False):
        st.markdown("**Preview how your CSV data will be mapped to RevyOps:**")
        
        # Show mapping for first row as example
        if len(normalized_df) > 0:
            from revyops_integration import map_csv_row_to_revyops, CUSTOM_FIELD_MAPPING
            sample_row = normalized_df.iloc[0]
            mapped_data = map_csv_row_to_revyops(sample_row)
            
            st.json(mapped_data)
            st.caption(f"📊 This is how the first row (Domain: {sample_row.get('Domain', 'N/A')}) will be sent to RevyOps")
            
            # Show which fields are mapped
            st.markdown("**Field Mapping Summary:**")
            mapped_fields = []
            unmapped_fields = []
            
            for col in normalized_df.columns:
                if col in ["Domain", "Name"]:
                    mapped_fields.append(f"✅ {col} → {col.lower()}")
                elif col in CUSTOM_FIELD_MAPPING:
                    mapped_fields.append(f"✅ {col} → {CUSTOM_FIELD_MAPPING[col]}")
                else:
                    unmapped_fields.append(f"⚠️ {col} (not in RevyOps schema)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mapped Fields:**")
                for field in mapped_fields[:15]:  # Show first 15
                    st.markdown(f"- {field}")
                if len(mapped_fields) > 15:
                    st.caption(f"... and {len(mapped_fields) - 15} more")
            
            with col2:
                if unmapped_fields:
                    st.markdown("**Unmapped Fields:**")
                    for field in unmapped_fields[:10]:  # Show first 10
                        st.markdown(f"- {field}")
                    if len(unmapped_fields) > 10:
                        st.caption(f"... and {len(unmapped_fields) - 10} more")
                else:
                    st.success("✅ All fields are mapped!")
        else:
            st.warning("No data to preview")
    
    # Get API key from session state
    current_api_key = st.session_state.get('revyops_api_key', '')
    
    if not current_api_key.strip():
        st.warning("⚠️ Enter your RevyOps API key above to push data to RevyOps.")
    else:
        # Show validation info before push
        valid_rows = normalized_df[
            (normalized_df["Domain"].notna()) & 
            (normalized_df["Domain"].astype(str).str.strip() != "") &
            (normalized_df["Name"].notna()) & 
            (normalized_df["Name"].astype(str).str.strip() != "")
        ]
        invalid_count = len(normalized_df) - len(valid_rows)
        
        if invalid_count > 0:
            st.warning(f"⚠️ {invalid_count} rows are missing required fields (Domain or Name) and will be skipped.")
        
        st.info(f"📊 Ready to push **{len(valid_rows)}** companies to RevyOps (out of {len(normalized_df)} total rows)")
        
        if st.button("🚀 Push to RevyOps", type="primary", use_container_width=True, key="push_button"):
            # Show immediate feedback
            status_container = st.empty()
            status_container.info("🔄 Starting push to RevyOps...")
            
            with st.spinner("🔄 Pushing companies to RevyOps..."):
                try:
                    # Progress callback function
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(processed, total, status):
                        progress = processed / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    # Push to RevyOps
                    summary = None
                    try:
                        status_container.info("🔄 Calling RevyOps API...")
                        # Use API key from session state
                        api_key_to_use = st.session_state.get('revyops_api_key', current_api_key).strip()
                        summary = push_companies_to_revyops(
                            normalized_df,
                            api_key_to_use,
                            progress_callback=update_progress
                        )
                        
                        # Verify summary was returned
                        if summary is None:
                            raise Exception("Push function returned None - no summary received")
                        
                        status_container.success("✅ Push operation completed!")
                        
                    except Exception as push_error:
                        status_container.error(f"❌ Error during push operation: {str(push_error)}")
                        st.error(f"❌ Error during push operation: {str(push_error)}")
                        st.exception(push_error)
                        summary = {
                            "success": 0,
                            "updated": 0,
                            "failed": len(normalized_df),
                            "errors": [{"row": 0, "domain": "N/A", "error": str(push_error)}],
                            "created_companies": [],
                            "updated_companies": []
                        }
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Verify we have a summary
                    if summary is None:
                        st.error("❌ No summary received from push operation. Please check the error logs.")
                        st.stop()
                    
                    # Store summary in session state for verification and persistence
                    st.session_state['revyops_summary'] = summary
                    st.session_state['revyops_push_complete'] = True
                    
                    # Force display - clear any previous messages
                    status_container.empty()
                    
                    # Display results immediately
                    st.markdown("---")
                    st.success(f"✅ **RevyOps Push Complete!**")
                    
                    # Always show summary - make it very visible
                    total_processed = summary.get('success', 0) + summary.get('updated', 0) + summary.get('failed', 0)
                    
                    st.markdown("### 📊 Push Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        created_count = summary.get('success', 0)
                        st.metric("✅ Created", created_count, delta=None)
                    with col2:
                        updated_count = summary.get('updated', 0)
                        st.metric("🔄 Updated", updated_count, delta=None)
                    with col3:
                        failed_count = summary.get('failed', 0)
                        st.metric("❌ Failed", failed_count, delta=None)
                    
                    st.markdown(f"**📈 Total Processed:** {total_processed} companies")
                    
                    # Debug info if nothing was processed
                    if total_processed == 0:
                        st.error("⚠️ **Warning:** No companies were processed. This might indicate an issue with the API connection or data format.")
                        st.info("💡 **Debug Info - Summary received:**")
                        st.json(summary)
                        
                        # Show first few rows of data for debugging
                        st.info("💡 **First 3 rows of data being sent:**")
                        if len(normalized_df) > 0:
                            for i in range(min(3, len(normalized_df))):
                                row = normalized_df.iloc[i]
                                st.code(f"Row {i+1}: Domain='{row.get('Domain', 'N/A')}', Name='{row.get('Name', 'N/A')}'")
                    
                    # Show detailed results - expand by default if there are results
                    if summary.get('created_companies'):
                        with st.expander(f"✅ Created Companies ({len(summary['created_companies'])})", expanded=True):
                            for company in summary['created_companies'][:50]:
                                company_id_display = f" (ID: {company.get('company_id', 'N/A')})" if company.get('company_id') else ""
                                st.success(f"Row {company['row']}: **{company['domain']}** - {company['name']}{company_id_display}")
                            if len(summary['created_companies']) > 50:
                                st.caption(f"... and {len(summary['created_companies']) - 50} more")
                    else:
                        st.info("ℹ️ No companies were created (they may have already existed)")
                    
                    if summary.get('updated_companies'):
                        with st.expander(f"🔄 Updated Companies ({len(summary['updated_companies'])})", expanded=True):
                            for company in summary['updated_companies'][:50]:
                                st.info(f"Row {company['row']}: **{company['domain']}** - {company['name']} (ID: {company.get('company_id', 'N/A')})")
                            if len(summary['updated_companies']) > 50:
                                st.caption(f"... and {len(summary['updated_companies']) - 50} more")
                    else:
                        st.info("ℹ️ No companies were updated")
                    
                    # Show errors if any
                    if summary.get('errors'):
                        with st.expander(f"❌ Error Details ({len(summary['errors'])})", expanded=True):
                            for error in summary['errors'][:20]:  # Show first 20 errors
                                st.error(f"**Row {error['row']}** (Domain: `{error['domain']}`): {error['error']}")
                            if len(summary['errors']) > 20:
                                st.caption(f"... and {len(summary['errors']) - 20} more errors")
                    else:
                        st.success("✅ No errors occurred!")
                    
                    # Verification button
                    st.divider()
                    if st.button("🔍 Verify Companies in RevyOps", use_container_width=True, key="verify_button_inline"):
                        with st.spinner("🔄 Verifying companies in RevyOps..."):
                            try:
                                # Get all domains that were created or updated
                                all_domains = []
                                if summary.get('created_companies'):
                                    all_domains.extend([c['domain'] for c in summary['created_companies']])
                                if summary.get('updated_companies'):
                                    all_domains.extend([c['domain'] for c in summary['updated_companies']])
                                
                                if all_domains:
                                    verification = verify_companies_in_revyops(all_domains, current_api_key.strip())
                                    
                                    st.subheader("🔍 Verification Results")
                                    
                                    if verification['found']:
                                        st.success(f"✅ Found {len(verification['found'])} companies in RevyOps")
                                        with st.expander(f"Found Companies ({len(verification['found'])})", expanded=True):
                                            for company in verification['found'][:30]:
                                                st.markdown(f"**{company['domain']}** - {company['name']}")
                                                st.caption(f"ID: {company['company_id']} | Status: {company['company_status']} | Updated: {company['updated_time']}")
                                            if len(verification['found']) > 30:
                                                st.caption(f"... and {len(verification['found']) - 30} more")
                                    
                                    if verification['not_found']:
                                        st.warning(f"⚠️ {len(verification['not_found'])} companies not found in RevyOps")
                                        with st.expander(f"Not Found ({len(verification['not_found'])})", expanded=False):
                                            for domain in verification['not_found'][:20]:
                                                st.error(domain)
                                            if len(verification['not_found']) > 20:
                                                st.caption(f"... and {len(verification['not_found']) - 20} more")
                                    
                                    if verification['errors']:
                                        st.error(f"❌ {len(verification['errors'])} verification errors")
                                        for error in verification['errors']:
                                            st.error(f"{error['domain']}: {error['error']}")
                                else:
                                    st.warning("No companies to verify (all failed or no data)")
                                    
                            except Exception as e:
                                st.error(f"❌ Error verifying companies: {str(e)}")
                                st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error pushing to RevyOps: {str(e)}")
                    st.exception(e)
                    st.session_state['revyops_push_error'] = str(e)

# Display RevyOps push results if they exist (persists across reruns)
if st.session_state.get('revyops_push_complete') and 'revyops_summary' in st.session_state:
    summary = st.session_state['revyops_summary']
    
    st.markdown("---")
    st.success(f"✅ **RevyOps Push Complete!**")
    
    # Always show summary - make it very visible
    total_processed = summary.get('success', 0) + summary.get('updated', 0) + summary.get('failed', 0)
    
    st.markdown("### 📊 Push Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        created_count = summary.get('success', 0)
        st.metric("✅ Created", created_count, delta=None)
    with col2:
        updated_count = summary.get('updated', 0)
        st.metric("🔄 Updated", updated_count, delta=None)
    with col3:
        failed_count = summary.get('failed', 0)
        st.metric("❌ Failed", failed_count, delta=None)
    
    st.markdown(f"**📈 Total Processed:** {total_processed} companies")
    
    # Debug info if nothing was processed
    if total_processed == 0:
        st.error("⚠️ **Warning:** No companies were processed. This might indicate an issue with the API connection or data format.")
        st.info("💡 **Debug Info - Summary received:**")
        st.json(summary)
        
        # Show first few rows of data for debugging
        if 'normalized_df' in st.session_state:
            normalized_df = st.session_state['normalized_df']
            st.info("💡 **First 3 rows of data being sent:**")
            if len(normalized_df) > 0:
                for i in range(min(3, len(normalized_df))):
                    row = normalized_df.iloc[i]
                    st.code(f"Row {i+1}: Domain='{row.get('Domain', 'N/A')}', Name='{row.get('Name', 'N/A')}'")
    
    # Show detailed results - expand by default if there are results
    if summary.get('created_companies'):
        with st.expander(f"✅ Created Companies ({len(summary['created_companies'])})", expanded=True):
            for company in summary['created_companies'][:50]:
                company_id_display = f" (ID: {company.get('company_id', 'N/A')})" if company.get('company_id') else ""
                st.success(f"Row {company['row']}: **{company['domain']}** - {company['name']}{company_id_display}")
            if len(summary['created_companies']) > 50:
                st.caption(f"... and {len(summary['created_companies']) - 50} more")
    else:
        st.info("ℹ️ No companies were created (they may have already existed)")
    
    if summary.get('updated_companies'):
        with st.expander(f"🔄 Updated Companies ({len(summary['updated_companies'])})", expanded=True):
            for company in summary['updated_companies'][:50]:
                st.info(f"Row {company['row']}: **{company['domain']}** - {company['name']} (ID: {company.get('company_id', 'N/A')})")
            if len(summary['updated_companies']) > 50:
                st.caption(f"... and {len(summary['updated_companies']) - 50} more")
    else:
        st.info("ℹ️ No companies were updated")
    
    # Show errors if any
    if summary.get('errors'):
        with st.expander(f"❌ Error Details ({len(summary['errors'])})", expanded=True):
            for error in summary['errors'][:20]:  # Show first 20 errors
                st.error(f"**Row {error['row']}** (Domain: `{error['domain']}`): {error['error']}")
            if len(summary['errors']) > 20:
                st.caption(f"... and {len(summary['errors']) - 20} more errors")
    else:
        st.success("✅ No errors occurred!")
    
    # Verification button
    st.divider()
    if st.button("🔍 Verify Companies in RevyOps", use_container_width=True, key="verify_button"):
        if 'normalized_df' in st.session_state and 'revyops_api_key' in st.session_state:
            revyops_api_key = st.session_state.get('revyops_api_key', '')
            with st.spinner("🔄 Verifying companies in RevyOps..."):
                try:
                    # Get all domains that were created or updated
                    all_domains = []
                    if summary.get('created_companies'):
                        all_domains.extend([c['domain'] for c in summary['created_companies']])
                    if summary.get('updated_companies'):
                        all_domains.extend([c['domain'] for c in summary['updated_companies']])
                    
                    if all_domains:
                        verification = verify_companies_in_revyops(all_domains, revyops_api_key.strip())
                        
                        st.subheader("🔍 Verification Results")
                        
                        if verification['found']:
                            st.success(f"✅ Found {len(verification['found'])} companies in RevyOps")
                            with st.expander(f"Found Companies ({len(verification['found'])})", expanded=True):
                                for company in verification['found'][:30]:
                                    st.markdown(f"**{company['domain']}** - {company['name']}")
                                    st.caption(f"ID: {company['company_id']} | Status: {company['company_status']} | Updated: {company['updated_time']}")
                                if len(verification['found']) > 30:
                                    st.caption(f"... and {len(verification['found']) - 30} more")
                        
                        if verification['not_found']:
                            st.warning(f"⚠️ {len(verification['not_found'])} companies not found in RevyOps")
                            with st.expander(f"Not Found ({len(verification['not_found'])})", expanded=False):
                                for domain in verification['not_found'][:20]:
                                    st.error(domain)
                                if len(verification['not_found']) > 20:
                                    st.caption(f"... and {len(verification['not_found']) - 20} more")
                        
                        if verification['errors']:
                            st.error(f"❌ {len(verification['errors'])} verification errors")
                            for error in verification['errors']:
                                st.error(f"{error['domain']}: {error['error']}")
                    else:
                        st.warning("No companies to verify (all failed or no data)")
                        
                except Exception as e:
                    st.error(f"❌ Error verifying companies: {str(e)}")
                    st.exception(e)
        else:
            st.error("❌ Missing data. Please process files first.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload Files**: Upload CSV files from any of the supported sources:
       - Crunchbase
       - Discolikes
       - AIARK
       - Apollo
       - LinkedIn
       - Google Maps
       
       You can upload one or more files - all uploaded files will be merged.
    
    2. **Client Name**: Enter the client name (e.g., "Buzzlead"). A folder will be created automatically.
    
    3. **Vertical**: Enter the vertical name (e.g., "Screen Print Shops" or "PR Firms").
    
    4. **Date**: Enter the date (e.g., "12-23-25").
    
    5. **Process**: Click the "Normalize & Merge Files" button
    
    6. **Download**: Once processing is complete, download your normalized CSV file
    
    **File Organization:**
    - **First Run**: Input files are automatically saved to `[Client Name]/Inputs/`
    - **Subsequent Runs**: Input files are saved to `[Client Name]/[Client Name] [Vertical] [Date] Draft/` subfolder
    - Output files are always saved to `[Client Name]/[Client Name] [Vertical] Accounts [Date].csv` (client root folder)
    - Folders are created automatically - no manual setup required!
    - You can upload files directly from your downloads folder
    
    The script will:
    - Map columns to canonical schema using the mapping configuration
    - Extract social media URLs (LinkedIn, Facebook, Twitter) where available
    - Create Website URLs from domains
    - Merge and deduplicate records from all sources
    - Output a normalized CSV file with all data combined
    
    **RevyOps Integration:**
    - After generating the CSV, you can push data directly to RevyOps
    - API keys are stored in `revyops_keys.json` (mapped by workspace name)
    - **Dedicated Workspaces:**
      - "Buzzlead" → Buzzlead workspace
      - "Product EVO" or "EVO" → Product EVO workspace
      - "Incentive Solutions Corp" or "Incentive Solutions" → Incentive Solutions Corp workspace
      - "Interdependence" or "IDPR" → Interdependence workspace
    - **Shared Workspace:**
      - All other clients → Master Account workspace
    - The system automatically selects the correct workspace and API key based on the Client Name
    - You can also set a default key in `.env` as `REVYOPS_API_KEY` as a fallback
    """)

