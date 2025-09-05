import os
import time
import argparse
import logging
import json
import csv
import threading
from datetime import datetime
from itertools import cycle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from dotenv import load_dotenv

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
load_dotenv()

class Config:
    """Loads and validates all necessary environment variables."""
    def __init__(self):
        self.gemini_api_keys = os.getenv('GEMINI_API_KEYS')
        self.gsheets_creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH')
        self.gsheets_spreadsheet_id = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
        if not self.gemini_api_keys:
            raise ValueError("GEMINI_API_KEYS environment variable is not set or is empty.")

class GeminiAPIKeyManager:
    """Thread-safe management of a pool of Gemini API keys."""
    def __init__(self, api_keys_str):
        self._keys = [key.strip() for key in api_keys_str.split(',')]
        self._key_pool = cycle(self._keys)
        self.exhausted_keys = set()
        self.lock = threading.Lock()
        with self.lock:
            self.current_key = next(self._key_pool, None)
        logging.info(f"Initialized with {len(self._keys)} API key(s).")

    def get_key(self):
        with self.lock:
            return self.current_key

    def rotate_key(self):
        """Rotates to the next available API key, returns False if all are exhausted."""
        with self.lock:
            if self.current_key:
                self.exhausted_keys.add(self.current_key)
                logging.warning(f"API Key ...{self.current_key[-4:]} marked as exhausted.")

            if len(self.exhausted_keys) >= len(self._keys):
                logging.error("All API keys have been exhausted.")
                return False

            while True:
                next_key = next(self._key_pool)
                if next_key not in self.exhausted_keys:
                    self.current_key = next_key
                    logging.info(f"Rotated to new API Key ...{self.current_key[-4:]}.")
                    return True
        return False

class DataManager:
    """Handles data I/O, gracefully failing over from Google Sheets to local CSV."""
    def __init__(self, config):
        self.mode = 'Google Sheets'
        self.sheet = self._connect_gsheets(config)
        self.csv_lock = threading.Lock()

        if not self.sheet:
            self.mode = 'CSV'
            logging.warning("Could not connect to Google Sheets. Switching to local CSV mode.")
            self.output_folder = Path("lead_results")
            self.output_folder.mkdir(exist_ok=True)
            self.output_file = self.output_folder / "lead_results.csv"
            self._initialize_csv()

    def _connect_gsheets(self, config):
        # Omitted for brevity, same as previous version
        if not config.gsheets_creds_path or not config.gsheets_spreadsheet_id:
            logging.warning("Google Sheets credentials or Spreadsheet ID not found in env.")
            return None
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(config.gsheets_creds_path, scopes=scopes)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(config.gsheets_spreadsheet_id)
            logging.info(f"Successfully connected to Google Sheet: {spreadsheet.title}")
            return spreadsheet.sheet1
        except Exception as e:
            logging.error(f"Failed to connect to Google Sheets: {e}")
            return None

    def _initialize_csv(self):
        with self.csv_lock:
            if not self.output_file.exists():
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'CompanyName', 'Status', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber',
                        'LinkedIn', 'CompanySummary', 'ReasonForQuality', 'Timestamp'
                    ])

    def get_all_company_names(self):
        """Fetches all company names from the data source for de-duplication."""
        names = set()
        try:
            if self.mode == 'Google Sheets':
                # Fetching only the first column for efficiency
                all_names = self.sheet.col_values(1)
                names = set(all_names[1:]) # Skip header
            else: # CSV Mode
                with self.csv_lock:
                    if not self.output_file.exists(): return set()
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        names = {row['CompanyName'] for row in reader if 'CompanyName' in row}
        except Exception as e:
            logging.error(f"Could not retrieve existing company names: {e}")
        return names

    def write_lead_data(self, company_name, lead_data):
        """Writes the generated lead data back to the appropriate destination."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        status = 'Processed' if lead_data.get('status') == 'success' else 'Failed'

        # Helper to return '' if data is None
        get = lambda key: lead_data.get(key) or ""

        if self.mode == 'Google Sheets':
            try:
                row_to_append = [
                    company_name, status, get('contact_name'), get('contact_title'), get('email'),
                    get('phone_number'), get('linkedin_url'), get('company_summary'),
                    get('reason_for_quality'), timestamp
                ]
                self.sheet.append_row(row_to_append, value_input_option='USER_ENTERED')
                logging.info(f"Appended sheet for '{company_name}' with status: {status}")
            except Exception as e:
                logging.error(f"Failed to update sheet for '{company_name}': {e}")
        else: # CSV Mode
            with self.csv_lock:
                try:
                    with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            company_name, status, get('contact_name'), get('contact_title'), get('email'),
                            get('phone_number'), get('linkedin_url'), get('company_summary'),
                            get('reason_for_quality'), timestamp
                        ])
                except Exception as e:
                    logging.error(f"Failed to write to CSV for '{company_name}': {e}")


class LeadGenerationOrchestrator:
    """Manages the entire two-phase lead generation process."""
    def __init__(self, config, args):
        self.key_manager = GeminiAPIKeyManager(config.gemini_api_keys)
        self.data_manager = DataManager(config)
        self.args = args

    def _call_gemini_api(self, prompt, is_json_output=True):
        """A robust, thread-safe method to call the Gemini API with retries and key rotation."""
        max_retries_per_key = 3
        while True: # Loop for key rotation
            api_key = self.key_manager.get_key()
            if not api_key:
                return None, "All API keys are exhausted."

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')

            for attempt in range(max_retries_per_key):
                try:
                    response = model.generate_content(prompt)
                    return response.text, None
                except ResourceExhausted:
                    wait_time = 2 ** attempt
                    logging.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                except Exception as e:
                    logging.error(f"An unexpected API error occurred: {e}")
                    return None, str(e)

            # If all retries fail, rotate key and try again
            if not self.key_manager.rotate_key():
                return None, "All API keys exhausted after retries."

    def _generate_company_list(self):
        """Phase 1: Generate a list of prospective companies."""
        logging.info("Phase 1: Generating company list...")
        prompt = (
            f"Act as a Market Research Analyst. Your task is to generate a list of exactly {self.args.num_companies} "
            f"company names that are strong potential leads in the '{self.args.sector}' sector, "
            f"focusing on the '{self.args.location}' area. "
            "Provide the output as a single line of comma-separated values. For example: 'Company A, Company B, Company C'."
        )
        response_text, error = self._call_gemini_api(prompt, is_json_output=False)
        if error:
            logging.error(f"Could not generate company list: {error}")
            return []
        
        companies = [name.strip() for name in response_text.split(',') if name.strip()]
        logging.info(f"Generated {len(companies)} company candidates.")
        return companies

    def _process_single_company(self, company_name):
        """Phase 2: Generate detailed lead info for a single company."""
        prompt = self._generate_detailed_prompt(company_name)
        raw_response, error = self._call_gemini_api(prompt)

        lead_data = {}
        if error:
            lead_data = {'status': 'failed', 'reason_for_quality': f"API Error: {error}"}
        else:
            try:
                # Clean the response before parsing
                clean_response = raw_response.strip().replace("```json", "").replace("```", "")
                parsed_json = json.loads(clean_response)
                lead_data = {
                    'status': 'success',
                    'contact_name': parsed_json.get('contact_person', {}).get('name'),
                    'contact_title': parsed_json.get('contact_person', {}).get('title'),
                    'email': parsed_json.get('contact_details', {}).get('email'),
                    'phone_number': parsed_json.get('contact_details', {}).get('phone_number'),
                    'linkedin_url': parsed_json.get('contact_details', {}).get('linkedin_url'),
                    'company_summary': parsed_json.get('company_summary'),
                    'reason_for_quality': parsed_json.get('reason_for_quality'),
                }
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON for '{company_name}'. Response: {raw_response[:200]}")
                lead_data = {'status': 'failed', 'reason_for_quality': "Invalid JSON response."}
        
        self.data_manager.write_lead_data(company_name, lead_data)
        return company_name

    def _generate_detailed_prompt(self, company_name):
        return f"""
        Act as an Expert Business Development Researcher. Your task is to find the most relevant decision-maker for a potential business partnership for the company specified below.

        ### COMPANY INFORMATION ###
        - Company Name: "{company_name}"
        - Geographic Focus: "{self.args.location}"
        - Industry Sector Focus: "{self.args.sector}"

        ### INSTRUCTIONS ###
        1. Find the best contact person (e.g., Manager, Director, VP in Marketing, Sales, or Business Development).
        2. Provide contact details, a company summary, and a reason for lead quality.
        3. Respond ONLY with a single, valid JSON object. Do not include any extra text or markdown.

        ### JSON OUTPUT FORMAT EXAMPLE ###
        {{
          "contact_person": {{
            "name": "Jane Doe",
            "title": "Director of Marketing"
          }},
          "contact_details": {{
            "email": "jane.doe@examplecorp.com",
            "phone_number": "+1-555-123-4567",
            "linkedin_url": "https://linkedin.com/in/janedoe"
          }},
          "company_summary": "Example Corp is a leading provider of cloud solutions.",
          "reason_for_quality": "As Director of Marketing, Jane is the key decision-maker for new tools."
        }}

        ### IMPORTANT ###
        If you cannot find specific information (like email or phone), use `null` as the value for that field. Do not invent information.
        """

    def run(self):
        """Executes the entire lead generation workflow."""
        # Phase 1: Generate and de-duplicate company list
        new_companies = self._generate_company_list()
        if not new_companies:
            return
            
        existing_companies = self.data_manager.get_all_company_names()
        companies_to_process = [c for c in new_companies if c not in existing_companies]

        if not companies_to_process:
            logging.info("All generated companies already exist in the database. Nothing new to process.")
            return

        logging.info(f"Starting detailed lead generation for {len(companies_to_process)} new companies...")

        # Phase 2: Process companies in parallel
        with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix='LeadGen') as executor:
            futures = []
            for company in companies_to_process:
                futures.append(executor.submit(self._process_single_company, company))
                time.sleep(1) # Stagger API requests to avoid initial burst
            
            for future in as_completed(futures):
                try:
                    company_name = future.result()
                    logging.info(f"Completed processing for: {company_name}")
                except Exception as e:
                    logging.error(f"A task generated an exception: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Advanced lead generation using a two-phase, parallel approach.")
    parser.add_argument('--location', type=str, required=True, help="Target location for leads (e.g., 'New York').")
    parser.add_argument('--sector', type=str, required=True, help="Industry sector for leads (e.g., 'Technology').")
    parser.add_argument('--num_companies', type=int, default=10, help="Number of companies to generate in the discovery phase.")
    parser.add_argument('--workers', type=int, default=5, help="Number of parallel threads for processing companies.")
    args = parser.parse_args()

    logging.info("Starting Advanced Lead Generation Script...")
    try:
        config = Config()
        orchestrator = LeadGenerationOrchestrator(config, args)
        orchestrator.run()
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
    except Exception as e:
        logging.error(f"A critical, unhandled error occurred: {e}", exc_info=True)

    logging.info("Lead Generation Script finished.")

if __name__ == "__main__":
    main()
