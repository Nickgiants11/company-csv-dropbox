#!/usr/bin/env python3
"""
V0 CSV normalizer/merger for two sources: Crunchbase + Discolikes.

Inputs (place files here):
- input/crunchbase.csv
- input/discolikes.csv

Mapping + schema:
- mappings/canonical_columns.json
- mappings/crunchbase_map.json
- mappings/discolikes_map.json

Outputs:
- output/companies_normalized.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
MAPPINGS_DIR = ROOT / "mappings"

CRUNCHBASE_CSV = INPUT_DIR / "crunchbase.csv"
DISCOLIKES_CSV = INPUT_DIR / "discolikes.csv"

CANONICAL_JSON = MAPPINGS_DIR / "canonical_columns.json"
MAPPINGS_CSV = MAPPINGS_DIR / "mapping.csv"

# Source file names
SOURCE_FILES = {
    "crunchbase": INPUT_DIR / "crunchbase.csv",
    "discolikes": INPUT_DIR / "discolikes.csv",
    "aiark": INPUT_DIR / "aiark.csv",
    "apollo": INPUT_DIR / "apollo.csv",
    "linkedin": INPUT_DIR / "linkedin.csv",
    "googlemaps": INPUT_DIR / "googlemaps.csv"
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mappings_from_csv() -> Dict[str, Dict[str, str]]:
    """
    Load mappings from CSV file and create mapping dictionaries for each source.
    Returns a dict with source names as keys and mapping dicts as values.
    """
    mappings_df = pd.read_csv(MAPPINGS_CSV)
    
    # Column names in the CSV
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
            
            # Skip empty mappings
            if pd.isna(row[source_col]) or source_header == "" or source_header == "nan":
                continue
            
            # Create mapping from source header to canonical column
            mapping[source_header] = canonical
        
        result[source_name] = mapping
    
    return result


def _get_source_value(row: pd.Series, source_name: str) -> str:
    """Get the Source value for a row based on the source name"""
    source_values = {
        "crunchbase": "Crunchbase",
        "discolikes": "Discolikes",
        "aiark": "AI-Ark",
        "apollo": "Apollo",
        "linkedin": "LinkedIn",
        "googlemaps": "Google Maps"
    }
    return source_values.get(source_name, source_name)


def _read_csv(path: Path) -> pd.DataFrame:
    # Tries to be tolerant of common CSV quirks
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")


def _normalize_domain(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.rstrip("/")
    # If a URL slipped in with a path, keep only host
    s = s.split("/")[0]
    return s.strip()


def _normalize_company_name(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_linkedin_url(val: str) -> str:
    """
    Extract LinkedIn URL from a string that may contain multiple URLs.
    Returns the first LinkedIn URL found, or empty string if none.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Pattern to match LinkedIn URLs (http/https, with or without www)
    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/[^\s,;|]+'
    match = re.search(linkedin_pattern, s, re.IGNORECASE)
    if match:
        return match.group(0)
    return ""


def _extract_facebook_url(val: str) -> str:
    """
    Extract Facebook URL from a string that may contain multiple URLs.
    Returns the first Facebook URL found, or empty string if none.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Pattern to match Facebook URLs (http/https, with or without www)
    facebook_pattern = r'https?://(?:www\.)?facebook\.com/[^\s,;|]+'
    match = re.search(facebook_pattern, s, re.IGNORECASE)
    if match:
        return match.group(0)
    return ""


def _extract_twitter_url(val: str) -> str:
    """
    Extract Twitter URL from a string that may contain multiple URLs.
    Returns the first Twitter URL found, or empty string if none.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Pattern to match Twitter URLs (http/https, with or without www)
    twitter_pattern = r'https?://(?:www\.)?twitter\.com/[^\s,;|]+'
    match = re.search(twitter_pattern, s, re.IGNORECASE)
    if match:
        return match.group(0)
    return ""


def _domain_to_website_url(val: str) -> str:
    """
    Convert domain to website URL by prepending https://
    Returns the full URL, or empty string if domain is empty.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Remove any existing protocol if present
    s = re.sub(r'^https?://', '', s)
    s = s.strip()
    if not s:
        return ""
    return f"https://{s}"


def _apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    # Map only columns that exist; ignore unmapped columns
    df2 = df.copy()
    
    # Group mappings by target column to handle duplicates
    target_to_sources = {}
    for src_col in df.columns:
        if src_col in mapping:
            target_col = mapping[src_col]
            if target_col not in target_to_sources:
                target_to_sources[target_col] = []
            target_to_sources[target_col].append(src_col)
    
    # Apply mappings, handling cases where multiple source columns map to same target
    for target_col, source_cols in target_to_sources.items():
        if len(source_cols) == 1:
            # Simple rename
            df2 = df2.rename(columns={source_cols[0]: target_col})
        else:
            # Multiple source columns map to same target - merge them
            # Keep the first one, append non-empty values from others
            primary_col = source_cols[0]
            df2 = df2.rename(columns={primary_col: target_col})
            
            # Merge values from other columns
            for other_col in source_cols[1:]:
                # Fill empty values in target with values from other column
                mask = (df2[target_col].isna() | (df2[target_col].astype(str).str.strip() == ""))
                df2.loc[mask, target_col] = df2.loc[mask, other_col]
                # Drop the other column
                df2 = df2.drop(columns=[other_col])
    
    return df2


def _ensure_canonical_columns(df: pd.DataFrame, canonical: List[str]) -> pd.DataFrame:
    for col in canonical:
        if col not in df.columns:
            df[col] = ""
    # Keep only canonical columns (in order)
    return df[canonical].copy()


def _row_completeness(row: pd.Series) -> int:
    # count non-empty fields
    return int(sum(1 for v in row.values.tolist() if isinstance(v, str) and v.strip() != ""))


def _dedupe_keep_most_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by:
    1) Domain (preferred)
    2) Name (fallback)
    Keep the row with the most filled-in fields.
    """
    # Build keys
    domains = df["Domain"].map(_normalize_domain) if "Domain" in df.columns else ""
    names = df["Name"].map(_normalize_company_name) if "Name" in df.columns else ""

    key = []
    for d, n in zip(domains.tolist(), names.tolist()):
        if d:
            key.append(f"domain:{d}")
        elif n:
            key.append(f"name:{n}")
        else:
            key.append("unknown:")
    df = df.copy()
    df["_dedupe_key"] = key

    # Sort by completeness descending so first row per key is best
    df["_completeness"] = df.apply(_row_completeness, axis=1)
    df = df.sort_values(by="_completeness", ascending=False)

    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first")
    df = df.drop(columns=["_dedupe_key", "_completeness"])
    return df


def _process_source(df: pd.DataFrame, source_name: str, mapping: Dict[str, str], canonical: List[str]) -> pd.DataFrame:
    """Process a single source dataframe with its mapping"""
    # Apply mapping
    df = _apply_mapping(df, mapping)
    
    # Source-specific processing
    if source_name == "crunchbase":
        # Auto-map website to domain for Crunchbase
        if "Website URL" in df.columns:
            df["Domain"] = df["Website URL"].map(_normalize_domain)
    
    elif source_name == "discolikes":
        # Extract social media URLs from Social URLs for Discolikes
        if "All Social URL's" in df.columns:
            df["LinkedIn URL"] = df["All Social URL's"].map(_extract_linkedin_url)
            df["Facebook URL"] = df["All Social URL's"].map(_extract_facebook_url)
            df["Twitter URL"] = df["All Social URL's"].map(_extract_twitter_url)
        # Auto-map domain to website URL for Discolikes
        if "Domain" in df.columns:
            df["Website URL"] = df["Domain"].map(_domain_to_website_url)
    
    elif source_name == "aiark":
        # Auto-map domain to website URL for AIARK
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
        # Auto-map domain to website URL for LinkedIn
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
    
    # Ensure canonical columns (this will add Source column if missing)
    df = _ensure_canonical_columns(df, canonical)
    
    # Normalize domains
    if "Domain" in df.columns:
        df["Domain"] = df["Domain"].map(_normalize_domain)
    
    # Set Source column value
    df["Source"] = _get_source_value(pd.Series(), source_name)
    
    return df


def main(output_filename: str | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if output_filename:
        # Ensure .csv extension
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        out_csv = OUTPUT_DIR / output_filename
    else:
        # Default filename with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = OUTPUT_DIR / f"companies_normalized_{timestamp}.csv"

    canonical = _load_json(CANONICAL_JSON)
    all_mappings = _load_mappings_from_csv()
    
    # Process all available sources
    processed_dfs = []
    
    for source_name, source_path in SOURCE_FILES.items():
        if not source_path.exists():
            print(f"⚠️  Skipping {source_name}: file not found at {source_path}")
            continue
        
        try:
            print(f"📂 Processing {source_name}...")
            df = _read_csv(source_path)
            # Remove "Tagline" and "Source" columns from input files
            if "Tagline" in df.columns:
                df = df.drop(columns=["Tagline"])
            if "Source" in df.columns:
                df = df.drop(columns=["Source"])
            mapping = all_mappings.get(source_name, {})
            
            if not mapping:
                print(f"⚠️  No mapping found for {source_name}, skipping...")
                continue
            
            processed_df = _process_source(df, source_name, mapping, canonical)
            processed_dfs.append(processed_df)
            print(f"✅ Processed {source_name}: {len(processed_df)} rows")
        except Exception as e:
            print(f"❌ Error processing {source_name}: {str(e)}")
            continue
    
    if not processed_dfs:
        raise ValueError("No valid source files found to process!")
    
    # Merge all dataframes
    merged = pd.concat(processed_dfs, ignore_index=True)
    merged = _dedupe_keep_most_complete(merged)

    merged.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Wrote: {out_csv} ({len(merged)} rows from {len(processed_dfs)} sources)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize and merge Crunchbase and Discolikes CSV files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (without .csv extension, will be added automatically). If not provided, uses timestamped filename."
    )
    args = parser.parse_args()
    main(output_filename=args.output)
