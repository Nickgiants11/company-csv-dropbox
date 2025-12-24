# CSV Normalizer

A tool to normalize and merge CSV files from Crunchbase and Discolikes sources.

## Features

- **Web Interface**: Easy-to-use Streamlit web interface for uploading and processing files
- **Column Mapping**: Automatically maps columns from both sources to a canonical schema
- **Social Media Extraction**: Extracts LinkedIn, Facebook, and Twitter URLs from social media fields
- **Domain Normalization**: Normalizes domains and creates website URLs
- **Deduplication**: Merges and deduplicates records, keeping the most complete entries
- **Custom Output Names**: Name your output files or use auto-timestamped filenames

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Upload your CSV files:
   - Upload Crunchbase CSV file
   - Upload Discolikes CSV file
   - (Optional) Enter a custom output filename
   - Click "Normalize & Merge Files"
   - Download your normalized CSV

### Command Line Interface

1. Place your CSV files in the `input/` folder:
   - `input/crunchbase.csv`
   - `input/discolikes.csv`

2. Run the normalization script:
```bash
# With custom output filename
python3 normalize.py -o my_output_name

# With auto-timestamped filename
python3 normalize.py
```

3. Find your output in the `output/` folder

## Project Structure

```
csv_normalizer_v0/
├── app.py                    # Streamlit web interface
├── normalize.py              # Main normalization script
├── requirements.txt          # Python dependencies
├── input/                    # Place input CSV files here
│   ├── crunchbase.csv
│   └── discolikes.csv
├── output/                   # Normalized output files
├── mappings/                 # Column mapping configurations
│   ├── canonical_columns.json
│   ├── crunchbase_map.json
│   └── discolikes_map.json
└── README.md
```

## What the Script Does

1. **Reads** both CSV files (Crunchbase and Discolikes)
2. **Maps** columns to canonical schema using mapping files
3. **Extracts** social media URLs (LinkedIn, Facebook, Twitter) from Discolikes
4. **Creates** Website URLs from domains for Discolikes
5. **Extracts** domains from Website URLs for Crunchbase
6. **Merges** both datasets
7. **Deduplicates** by domain (preferred) or name (fallback), keeping most complete records
8. **Outputs** a normalized CSV file

## Accessing from Other Devices

To access the web interface from other devices on your network:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then access it at `http://YOUR_IP_ADDRESS:8501` from any device on your network.

# company-csv-dropbox
