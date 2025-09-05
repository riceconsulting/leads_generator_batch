import os
import logging
import csv
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class Config:
    """Loads and validates environment variables required for Google Sheets sync."""
    def __init__(self):
        # Support for GitHub Actions where the JSON content is directly in the secret
        self.gsheets_creds_json = os.getenv('GOOGLE_SHEETS_CREDENTIALS_JSON')
        self.gsheets_creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
        self.gsheets_spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
        self.gsheets_sheet_name = os.getenv('GOOGLE_SHEETS_SHEET_NAME', 'list_of_leads') # Allow custom sheet name

        if not self.gsheets_spreadsheet_id:
            raise ValueError("GOOGLE_SHEETS_SPREADSHEET_ID is not set.")
        if not self.gsheets_creds_path and not self.gsheets_creds_json:
            raise ValueError("Either GOOGLE_SHEETS_CREDENTIALS_PATH or GOOGLE_SHEETS_CREDENTIALS_JSON must be set.")

def connect_to_google_sheets(config):
    """Connects to Google Sheets using service account credentials and returns the sheet object."""
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        if config.gsheets_creds_json:
            # Load credentials from JSON string (for GitHub Actions)
            import json
            creds_dict = json.loads(config.gsheets_creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        else:
            # Load credentials from file path (for local development)
            creds = Credentials.from_service_account_file(config.gsheets_creds_path, scopes=scopes)

        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(config.gsheets_spreadsheet_id)
        sheet = spreadsheet.worksheet(config.gsheets_sheet_name)
        logging.info(f"Successfully connected to Google Sheet: '{spreadsheet.title}' -> Worksheet: '{sheet.title}'")
        return sheet
    except gspread.exceptions.WorksheetNotFound:
        logging.error(f"Worksheet '{config.gsheets_sheet_name}' not found in the spreadsheet. Please create it.")
        return None
    except Exception as e:
        logging.error(f"Failed to connect to Google Sheets: {e}")
        return None

def main():
    """Main function to sync leads from local CSV to Google Sheets."""
    logging.info("Starting Google Sheets Sync Script...")
    try:
        config = Config()
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
        return

    sheet = connect_to_google_sheets(config)
    if not sheet:
        logging.error("Halting script due to connection failure.")
        return

    # 1. Read existing company names from the Google Sheet
    try:
        existing_sheet_companies = set(sheet.col_values(1)[1:]) # Get all of column A, skip header
        logging.info(f"Found {len(existing_sheet_companies)} existing companies in the sheet.")
    except Exception as e:
        logging.error(f"Could not fetch existing company names from the sheet: {e}")
        return
    
    # 2. Read all leads from the local CSV file
    csv_file_path = Path("lead_results/lead_results.csv")
    if not csv_file_path.exists():
        logging.warning("lead_results.csv not found. Nothing to sync.")
        return

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            csv_leads = list(csv.DictReader(f))
        logging.info(f"Read {len(csv_leads)} leads from {csv_file_path}.")
    except Exception as e:
        logging.error(f"Could not read the CSV file: {e}")
        return

    # 3. Identify new leads that are not in the sheet
    leads_to_append = []
    headers = ['CompanyName', 'Status', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber',
               'LinkedIn', 'CompanySummary', 'ReasonForQuality', 'Timestamp']

    for lead in csv_leads:
        company_name = lead.get('CompanyName')
        if company_name and company_name not in existing_sheet_companies:
            # Create the row in the correct order based on headers
            new_row = [lead.get(header, "") for header in headers]
            leads_to_append.append(new_row)
            # Add to set to avoid trying to add duplicates from the CSV in the same run
            existing_sheet_companies.add(company_name) 

    # 4. Batch append the new leads to the sheet
    if not leads_to_append:
        logging.info("No new leads to sync. Google Sheet is already up-to-date.")
    else:
        logging.info(f"Found {len(leads_to_append)} new leads to append to the sheet.")
        try:
            sheet.append_rows(leads_to_append, value_input_option='USER_ENTERED')
            logging.info("Successfully synced new leads to Google Sheets.")
        except Exception as e:
            logging.error(f"An error occurred while appending rows to the sheet: {e}")

    logging.info("Google Sheets Sync Script finished.")

if __name__ == "__main__":
    main()
