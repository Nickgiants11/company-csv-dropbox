"""
RevyOps API Integration Module
Handles pushing company data from CSV to RevyOps.com API
"""
import time
import pandas as pd
import requests
from typing import Dict, List, Tuple, Optional, Callable


REVYOPS_BASE_URL = "https://app.revyops.com/api"
REVYOPS_POST_ENDPOINT = f"{REVYOPS_BASE_URL}/public/companies"
REVYOPS_GET_ENDPOINT = f"{REVYOPS_BASE_URL}/public/companies"


# Column mapping: CSV column name -> RevyOps custom field name
CUSTOM_FIELD_MAPPING = {
    "LinkedIn URL": "LinkedIn URL",
    "Industry": "Industry",
    "Employee Count": "Employee Count",
    "Description": "Description",
    "Estimated Revenue": "Estimated Revenue",
    "Keywords": "Keywords",
    "Tagline": "Tagline",
    "Website URL": "Website URL",
    "Subindustry": "Subindustry",
    "City": "City",
    "State": "State",
    "Country": "Country",
    "Location": "Location",
    "Address": "Address",
    "Zip Code": "Zip Code",
    "Technologies Used": "Technologies Used",
    "Twitter URL": "Twitter URL",
    "Founded Date": "Founded Date",
    "Facebook URL": "Facebook URL",
    "All Social URL's": "ALL Social URL's",  # Note: exact match required
    "Phone Number": "Phone Number",
    "General Emails": "General Email(s)",  # Note: different name
    "IPO Status": "IPO Status",
    "Business Type": "Business Type",
    "Rating": "Rating",
    "Review Count": "Review Count",
    "Number of Locations": "Number of Locations",
    "Source": "Source",
    "Total Funding": "Total Funding",
    "Last Funding Date": "Last Funding Date",
    "Last Funding Amount": "Last Funding Amount",
    "Funding Stage": "Funding Stage",
    "Number of Founders": "Number of Founders",
    "Founders": "Founders",
    "Client": "Client",
}


def map_csv_row_to_revyops(row: pd.Series) -> dict:
    """
    Maps a CSV row to RevyOps JSON schema.
    
    Args:
        row: pandas Series representing a CSV row
        
    Returns:
        dict: RevyOps company data structure
    """
    # Direct mappings
    company_data = {
        "domain": str(row.get("Domain", "")).strip() if pd.notna(row.get("Domain")) else "",
        "name": str(row.get("Name", "")).strip() if pd.notna(row.get("Name")) else "",
        "company_status": "",  # Empty string as specified
        "company_custom_fields": []
    }
    
    # Map all custom fields
    for csv_column, field_name in CUSTOM_FIELD_MAPPING.items():
        if csv_column in row.index:
            value = row[csv_column]
            # Handle empty/NaN values
            if pd.notna(value) and str(value).strip():
                company_data["company_custom_fields"].append({
                    "field_name": field_name,
                    "field_value": str(value).strip()
                })
    
    return company_data


def post_company_to_revyops(company_data: dict, api_key: str, retries: int = 2) -> Tuple[bool, dict, int]:
    """
    Sends POST request to create a company in RevyOps.
    
    Args:
        company_data: Company data dictionary
        api_key: RevyOps API key
        retries: Number of retry attempts for network errors
        
    Returns:
        Tuple of (success: bool, response_data: dict, status_code: int)
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                REVYOPS_POST_ENDPOINT,
                json=company_data,
                headers=headers,
                timeout=30
            )
            
            status_code = response.status_code
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"message": response.text}
            
            # Success cases
            if status_code == 201:
                return (True, response_data, status_code)
            
            # Conflict - company already exists
            if status_code == 409:
                return (False, response_data, status_code)
            
            # Client errors - don't retry
            if status_code in [400, 403]:
                return (False, response_data, status_code)
            
            # Server errors - retry
            if status_code >= 500 and attempt < retries:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            
            return (False, response_data, status_code)
            
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
                continue
            return (False, {"error": str(e)}, 0)
    
    return (False, {"error": "Max retries exceeded"}, 0)


def get_company_by_domain(domain: str, api_key: str) -> Optional[dict]:
    """
    Gets a company by domain from RevyOps.
    
    Args:
        domain: Company domain
        api_key: RevyOps API key
        
    Returns:
        Company data dict if found, None otherwise
    """
    headers = {
        "x-api-key": api_key
    }
    
    try:
        response = requests.get(
            REVYOPS_GET_ENDPOINT,
            params={"domain": domain},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            companies = response.json()
            if companies and len(companies) > 0:
                return companies[0]  # Return first match
        return None
    except requests.exceptions.RequestException:
        return None


def patch_company_in_revyops(company_id: int, company_data: dict, api_key: str, retries: int = 2) -> Tuple[bool, dict, int]:
    """
    Sends PATCH request to update an existing company in RevyOps.
    
    Args:
        company_id: RevyOps company ID
        company_data: Company data dictionary
        api_key: RevyOps API key
        retries: Number of retry attempts for network errors
        
    Returns:
        Tuple of (success: bool, response_data: dict, status_code: int)
    """
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    patch_url = f"{REVYOPS_BASE_URL}/public/companies/{company_id}"
    
    for attempt in range(retries + 1):
        try:
            response = requests.patch(
                patch_url,
                json=company_data,
                headers=headers,
                timeout=30
            )
            
            status_code = response.status_code
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"message": response.text}
            
            # Success cases
            if status_code in [200, 204]:
                return (True, response_data, status_code)
            
            # Client errors - don't retry
            if status_code in [400, 403, 404]:
                return (False, response_data, status_code)
            
            # Server errors - retry
            if status_code >= 500 and attempt < retries:
                time.sleep(1 * (attempt + 1))
                continue
            
            return (False, response_data, status_code)
            
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
                continue
            return (False, {"error": str(e)}, 0)
    
    return (False, {"error": "Max retries exceeded"}, 0)


def push_companies_to_revyops(
    df: pd.DataFrame,
    api_key: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> dict:
    """
    Main function to push all companies from DataFrame to RevyOps.
    
    Args:
        df: DataFrame with company data
        api_key: RevyOps API key
        progress_callback: Optional callback function(processed, total, status)
        
    Returns:
        Summary dict with success, updated, failed counts and errors list
    """
    summary = {
        "success": 0,
        "updated": 0,
        "failed": 0,
        "errors": [],
        "created_companies": [],  # List of domains that were created
        "updated_companies": []   # List of domains that were updated
    }
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        # Update progress
        if progress_callback:
            progress_callback(idx + 1, total_rows, f"Processing row {idx + 1} of {total_rows}")
        
        # Map CSV row to RevyOps schema
        company_data = map_csv_row_to_revyops(row)
        
        # Validate required fields
        if not company_data.get("domain") or not company_data.get("name"):
            summary["failed"] += 1
            summary["errors"].append({
                "row": idx + 1,
                "domain": company_data.get("domain", "N/A"),
                "error": "Missing required field: domain or name"
            })
            continue
        
        # Try to POST company
        domain = company_data.get("domain", "N/A")
        company_name = company_data.get("name", "N/A")
        
        success, response_data, status_code = post_company_to_revyops(company_data, api_key)
        
        # Always try to get company_id and run PATCH, regardless of POST result
        company_id = None
        
        if success:
            # POST succeeded - extract company_id from response
            company_id = response_data.get("id") or response_data.get("company_id")
        
        # If we don't have company_id yet, lookup by domain
        if not company_id:
            existing_company = get_company_by_domain(domain, api_key)
            if existing_company and "id" in existing_company:
                company_id = existing_company["id"]
        
        # Always run PATCH if we have a company_id
        if company_id:
            patch_success, patch_response, patch_status = patch_company_in_revyops(
                company_id, company_data, api_key
            )
            
            if patch_success:
                if success:
                    # POST succeeded and PATCH succeeded - company was created and updated
                    summary["success"] += 1
                    summary["updated"] += 1
                    summary["created_companies"].append({
                        "domain": domain,
                        "name": company_name,
                        "row": idx + 1,
                        "company_id": company_id
                    })
                    summary["updated_companies"].append({
                        "domain": domain,
                        "name": company_name,
                        "row": idx + 1,
                        "company_id": company_id
                    })
                else:
                    # POST failed but PATCH succeeded - company was updated
                    summary["updated"] += 1
                    summary["updated_companies"].append({
                        "domain": domain,
                        "name": company_name,
                        "row": idx + 1,
                        "company_id": company_id
                    })
            else:
                # PATCH failed
                if success:
                    # POST succeeded but PATCH failed
                    summary["success"] += 1
                    summary["created_companies"].append({
                        "domain": domain,
                        "name": company_name,
                        "row": idx + 1,
                        "company_id": company_id
                    })
                    summary["failed"] += 1
                    summary["errors"].append({
                        "row": idx + 1,
                        "domain": domain,
                        "error": f"POST succeeded but PATCH failed: {patch_status} - {patch_response}"
                    })
                else:
                    # Both POST and PATCH failed
                    summary["failed"] += 1
                    error_msg = response_data.get("message", response_data.get("error", "Unknown error"))
                    summary["errors"].append({
                        "row": idx + 1,
                        "domain": domain,
                        "error": f"POST failed ({status_code}: {error_msg}) and PATCH failed: {patch_status} - {patch_response}"
                    })
        else:
            # No company_id available - can't run PATCH
            if success:
                # POST succeeded but couldn't get company_id for PATCH
                summary["success"] += 1
                summary["created_companies"].append({
                    "domain": domain,
                    "name": company_name,
                    "row": idx + 1,
                    "company_id": None
                })
                summary["failed"] += 1
                summary["errors"].append({
                    "row": idx + 1,
                    "domain": domain,
                    "error": "POST succeeded but couldn't retrieve company_id for PATCH update"
                })
            elif status_code == 409:
                # POST conflict but couldn't find company
                summary["failed"] += 1
                summary["errors"].append({
                    "row": idx + 1,
                    "domain": domain,
                    "error": f"Conflict (409) but couldn't retrieve company for update"
                })
            else:
                # POST failed and couldn't get company_id
                summary["failed"] += 1
                error_msg = response_data.get("message", response_data.get("error", "Unknown error"))
                summary["errors"].append({
                    "row": idx + 1,
                    "domain": company_data.get("domain", "N/A"),
                    "error": f"POST failed ({status_code}: {error_msg}) and couldn't retrieve company for PATCH"
                })
        
        # Rate limiting - small delay between requests
        time.sleep(0.5)
    
    return summary


def verify_companies_in_revyops(domains: List[str], api_key: str) -> dict:
    """
    Verify which companies exist in RevyOps by checking their domains.
    
    Args:
        domains: List of domain names to check
        api_key: RevyOps API key
        
    Returns:
        dict with found, not_found lists and details
    """
    verification_result = {
        "found": [],
        "not_found": [],
        "errors": []
    }
    
    headers = {
        "x-api-key": api_key
    }
    
    for domain in domains:
        if not domain or domain == "N/A":
            continue
            
        try:
            response = requests.get(
                REVYOPS_GET_ENDPOINT,
                params={"domain": domain},
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                companies = response.json()
                if companies and len(companies) > 0:
                    company = companies[0]
                    verification_result["found"].append({
                        "domain": domain,
                        "company_id": company.get("id"),
                        "name": company.get("name"),
                        "company_status": company.get("company_status"),
                        "updated_time": company.get("updated_time")
                    })
                else:
                    verification_result["not_found"].append(domain)
            elif response.status_code == 403:
                verification_result["errors"].append({
                    "domain": domain,
                    "error": "403 Forbidden - Invalid API key"
                })
            else:
                verification_result["not_found"].append(domain)
                
        except requests.exceptions.RequestException as e:
            verification_result["errors"].append({
                "domain": domain,
                "error": str(e)
            })
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
    
    return verification_result

