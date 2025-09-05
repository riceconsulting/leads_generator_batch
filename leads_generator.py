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

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
load_dotenv()

class Config:
    """Loads and validates all necessary environment variables for lead generation."""
    def __init__(self):
        self.gemini_api_keys = os.getenv('GEMINI_API_KEYS')
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

class CSVDataManager:
    """Handles all data I/O robustly with a local CSV file."""
    def __init__(self):
        self.csv_lock = threading.Lock()
        self.output_folder = Path("lead_results")
        self.output_folder.mkdir(exist_ok=True)
        self.output_file = self.output_folder / "lead_results.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        with self.csv_lock:
            if not self.output_file.exists():
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'CompanyName', 'Status', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber',
                        'LinkedIn', 'CompanySummary', 'ReasonForQuality', 'Timestamp'
                    ])

    def get_existing_company_names(self):
        with self.csv_lock:
            if not self.output_file.exists(): return set()
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return {row['CompanyName'] for row in reader if 'CompanyName' in row}
            except Exception as e:
                logging.error(f"Could not read existing company names from CSV: {e}")
                return set()

    def write_lead_data(self, company_name, lead_data):
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        status = 'Processed' if lead_data.get('status') == 'success' else 'Failed'
        get = lambda key: lead_data.get(key) or ""

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
    """Manages the entire two-phase lead generation process, saving results to CSV."""
    def __init__(self, config, args):
        self.key_manager = GeminiAPIKeyManager(config.gemini_api_keys)
        self.data_manager = CSVDataManager()
        self.args = args

    def _call_gemini_api(self, prompt):
        max_retries_per_key = 3
        while True:
            api_key = self.key_manager.get_key()
            if not api_key: return None, "All API keys are exhausted."
            genai.configure(api_key=api_key)
            
            # --- THIS IS THE REVISED LINE ---
            # Changed 'gemini-pro' to 'gemini-1.5-flash-latest' to align with modern, available models.
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            for attempt in range(max_retries_per_key):
                try:
                    response = model.generate_content(prompt)
                    return response.text, None
                except ResourceExhausted:
                    logging.warning(f"Rate limit hit. Retrying in {2**attempt}s...")
                    time.sleep(2 ** attempt)
                except Exception as e:
                    logging.error(f"An unexpected API error occurred: {e}")
                    return None, str(e)
            if not self.key_manager.rotate_key():
                return None, "All API keys exhausted after retries."

    def _generate_company_list_in_batches(self):
        logging.info("Phase 1: Generating company list in batches...")
        all_new_companies = set()
        num_batches = (self.args.num_companies + 9) // 10

        for i in range(num_batches):
            existing_companies = self.data_manager.get_existing_company_names().union(all_new_companies)
            logging.info(f"Batch {i+1}/{num_batches}: Generating 10 new companies...")
            avoid_list_str = ", ".join(list(existing_companies)[-200:])
            
            prompt = (
                f"Act as a Market Research Analyst. Your task is to generate a list of 10 high-quality company names "
                f"that are strong potential leads in the '{self.args.sector}' sector, focusing on the '{self.args.location}' area. "
                "Provide the output as a single line of comma-separated values (e.g., 'Company A, Company B, Company C').\n\n"
                f"IMPORTANT: Do NOT include any of the following company names in your output:\n{avoid_list_str}"
            )
            
            response_text, error = self._call_gemini_api(prompt)
            if error:
                logging.error(f"Could not generate company list for batch {i+1}: {error}")
                continue
            
            batch_companies = {name.strip() for name in response_text.split(',') if name.strip()}
            newly_found = batch_companies - existing_companies
            all_new_companies.update(newly_found)
            logging.info(f"Found {len(newly_found)} unique companies in this batch.")
            time.sleep(2)

        return list(all_new_companies)[:self.args.num_companies]

    def _process_single_company(self, company_name):
        prompt = self._generate_detailed_prompt(company_name)
        raw_response, error = self._call_gemini_api(prompt)
        lead_data = {}
        if error:
            lead_data = {'status': 'failed', 'reason_for_quality': f"API Error: {error}"}
        else:
            try:
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
        companies_to_process = self._generate_company_list_in_batches()
        if not companies_to_process:
            logging.warning("No new companies were generated to process.")
            return

        logging.info(f"Starting detailed lead generation for {len(companies_to_process)} new companies...")
        with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix='LeadGen') as executor:
            futures = [executor.submit(self._process_single_company, company) for company in companies_to_process]
            for future in as_completed(futures):
                try:
                    logging.info(f"Completed processing for: {future.result()}")
                except Exception as e:
                    logging.error(f"A task generated an exception: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Generates leads and saves them to a local CSV file.")
    parser.add_argument('--location', type=str, required=True, help="Target location for leads.")
    parser.add_argument('--sector', type=str, required=True, help="Industry sector for leads.")
    parser.add_argument('--num_companies', type=int, default=10, help="Total number of new companies to generate.")
    parser.add_argument('--workers', type=int, default=5, help="Number of parallel threads for processing.")
    args = parser.parse_args()

    logging.info("Starting Lead Generation Script...")
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
