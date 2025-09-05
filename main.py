import os
import time
import argparse
import logging
from datetime import datetime
from itertools import cycle

import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from dotenv import load_dotenv

# --- Configuration ---
# Set up basic logging to see the script's progress and any errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from a .env file
load_dotenv()

# --- Main Application Logic ---

class GeminiAPIKeyManager:
    """
    Manages a pool of Gemini API keys, allowing for rotation when a key is exhausted.
    """
    def __init__(self, api_keys_str):
        if not api_keys_str:
            raise ValueError("GEMINI_API_KEYS environment variable is not set.")
        self._keys = [key.strip() for key in api_keys_str.split(',')]
        self._key_pool = cycle(self._keys)
        self.current_key = next(self._key_pool, None)
        self.exhausted_keys = set()

    def get_key(self):
        """Returns the current active API key."""
        return self.current_key

    def rotate_key(self):
        """Rotates to the next available API key."""
        if self.current_key:
            self.exhausted_keys.add(self.current_key)
            logging.warning(f"API Key ending in '...{self.current_key[-4:]}' has been marked as exhausted.")

        if len(self.exhausted_keys) == len(self._keys):
            logging.error("All API keys have been exhausted.")
            return None

        # Find the next key that is not in the exhausted set
        while True:
            next_key = next(self._key_pool, None)
            if next_key not in self.exhausted_keys:
                self.current_key = next_key
                logging.info(f"Rotated to new API Key ending in '...{self.current_key[-4:]}'.")
                return self.current_key

def get_google_sheet_client():
    """
    Connects to the Google Sheets API using service account credentials.
    """
    try:
        creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
        spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')

        if not creds_path or not spreadsheet_id:
            raise ValueError("Google Sheets credentials path or Spreadsheet ID is not set in environment variables.")

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(spreadsheet_id)
        logging.info(f"Successfully connected to Google Sheet: {spreadsheet.title}")
        return spreadsheet.sheet1
    except Exception as e:
        logging.error(f"Failed to connect to Google Sheets: {e}")
        return None

def generate_lead_prompt(company_name, location=None, sector=None):
    """
    Creates a detailed prompt for the Gemini API to generate lead information.
    """
    prompt = (
        f"Generate high-quality business lead information for the company: '{company_name}'.\n"
        "Provide the output in a clean, parsable text format, with each piece of information on a new line, like this:\n"
        "Contact Person: [Name, Job Title]\n"
        "Email: [Email Address]\n"
        "Phone Number: [Phone Number]\n"
        "Company Summary: [A brief, insightful summary of the company]\n"
        "Reason for Lead Quality: [A short explanation of why this is a good lead]\n\n"
    )

    if location and sector:
        prompt += f"Focus specifically on contacts within the '{sector}' sector located in or near '{location}'.\n"
    elif location:
        prompt += f"Focus specifically on contacts located in or near '{location}'.\n"
    elif sector:
        prompt += f"Focus specifically on contacts within the '{sector}' sector.\n"

    prompt += "If you cannot find specific information for a field, please indicate with 'N/A'."
    return prompt

def process_companies(sheet, key_manager, location, sector):
    """
    Main function to fetch companies, generate leads, and update the sheet.
    """
    try:
        all_records = sheet.get_all_records()
    except gspread.exceptions.GSpreadException as e:
        logging.error(f"Could not fetch records from Google Sheet. Check permissions and sheet name. Error: {e}")
        return

    # Filter for companies that have not been processed yet
    # Assumes columns: 'CompanyName', 'Status', 'GeneratedLeadInfo', 'Timestamp'
    unprocessed_companies = [
        (i + 2, row) for i, row in enumerate(all_records)
        if 'Status' not in row or row.get('Status', '').lower() not in ['processed', 'failed']
    ]

    if not unprocessed_companies:
        logging.info("No new companies to process.")
        return

    logging.info(f"Found {len(unprocessed_companies)} companies to process.")

    for row_index, company_data in unprocessed_companies:
        company_name = company_data.get('CompanyName')
        if not company_name:
            logging.warning(f"Skipping row {row_index} due to missing company name.")
            continue

        logging.info(f"Processing '{company_name}' (Row {row_index})...")

        prompt = generate_lead_prompt(company_name, location, sector)
        max_retries_per_key = 3
        lead_info = None

        while True: # This loop handles key rotation
            api_key = key_manager.get_key()
            if not api_key:
                logging.error("Halting process: No more available API keys.")
                return # Exit the entire function

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')

            # Inner loop for exponential backoff retries for the *current* key
            for attempt in range(max_retries_per_key):
                try:
                    response = model.generate_content(prompt)
                    lead_info = response.text
                    logging.info(f"Successfully generated lead for '{company_name}'.")
                    break  # Exit retry loop on success
                except ResourceExhausted as e:
                    wait_time = 2 ** attempt
                    logging.warning(f"Rate limit hit for key ...{api_key[-4:]}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries_per_key})")
                    time.sleep(wait_time)
                except GoogleAPICallError as e:
                    logging.error(f"A Google API call error occurred for '{company_name}': {e}")
                    lead_info = f"Error: {e}"
                    break # Stop retrying on non-rate-limit API errors
                except Exception as e:
                    logging.error(f"An unexpected error occurred while processing '{company_name}': {e}")
                    lead_info = f"Error: {e}"
                    break # Stop retrying on other errors

            if lead_info: # If we got a result (or an error string)
                break # Exit the key rotation loop
            else: # If all retries failed for the current key
                if not key_manager.rotate_key():
                    return # Stop if no keys are left

        # Update the Google Sheet with the result
        try:
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            status = 'Processed' if not lead_info.startswith('Error:') else 'Failed'
            sheet.update(f'B{row_index}:D{row_index}', [[status, lead_info, timestamp]])
            logging.info(f"Updated sheet for '{company_name}' with status: {status}")
        except Exception as e:
            logging.error(f"Failed to update sheet for '{company_name}': {e}")

        time.sleep(2) # A small delay to be respectful to the APIs

def main():
    """
    Main entry point for the script. Parses arguments and starts the process.
    """
    parser = argparse.ArgumentParser(description="Generate business leads using Gemini API and Google Sheets.")
    parser.add_argument('--location', type=str, help="Optional: Specify the target location for leads (e.g., 'New York').")
    parser.add_argument('--sector', type=str, help="Optional: Specify the industry sector for leads (e.g., 'Technology').")
    args = parser.parse_args()

    logging.info("Starting Lead Generation Script...")
    if args.location:
        logging.info(f"Customization: Targeting location '{args.location}'")
    if args.sector:
        logging.info(f"Customization: Targeting sector '{args.sector}'")

    try:
        key_manager = GeminiAPIKeyManager(os.getenv('GEMINI_API_KEYS'))
        sheet = get_google_sheet_client()

        if sheet:
            process_companies(sheet, key_manager, args.location, args.sector)

    except ValueError as e:
        logging.error(f"Configuration error: {e}")
    except Exception as e:
        logging.error(f"A critical error occurred: {e}")

    logging.info("Lead Generation Script finished.")

if __name__ == "__main__":
    main()
