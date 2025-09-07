# File: leads_generator.py
# --- REVISED TO ADD 'COMPANYWEBSITE' COLUMN ---

import os
import time
import argparse
import logging
import json
import csv
import re
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
        """Initializes the CSV with the correct headers if it doesn't exist."""
        with self.csv_lock:
            if self.output_file.exists():
                return
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # --- REVISION: Added 'CompanyWebsite' column to the CSV header ---
                writer.writerow([
                    'CompanyName', 'Industry', 'CompanyWebsite', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber', 'Timestamp'
                ])

    def get_existing_company_names(self):
        """Reads all distinct company names from the CSV once for efficiency."""
        with self.csv_lock:
            if not self.output_file.exists(): return set()
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return {row['CompanyName'] for row in reader if 'CompanyName' in row}
            except Exception as e:
                logging.error(f"Could not read existing company names from CSV: {e}")
                return set()

    def write_lead_data(self, company_name, industry, website, contacts):
        """Writes multiple rows to the CSV, one for each valid contact found."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        with self.csv_lock:
            try:
                with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for contact in contacts:
                        # --- REVISION: Added 'website' to the row data ---
                        writer.writerow([
                            company_name,
                            industry,
                            website,
                            contact.get('name') or "",
                            contact.get('title') or "",
                            contact.get('email') or "",
                            contact.get('phone') or "",
                            timestamp
                        ])
            except Exception as e:
                logging.error(f"Failed to write to CSV for '{company_name}': {e}")


class LeadGenerationOrchestrator:
    """Manages the entire lead generation process with a regenerative failure loop."""
    def __init__(self, config, args):
        self.key_manager = GeminiAPIKeyManager(config.gemini_api_keys)
        self.data_manager = CSVDataManager()
        self.args = args

    def _call_gemini_api(self, prompt):
        """Calls Gemini API with retries and the Google Search tool."""
        max_retries_per_key = 5
        while True:
            api_key = self.key_manager.get_key()
            if not api_key: return None, "All API keys are exhausted."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash'
            )
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
            if not self.key_manager.rotate_key():
                return None, "All API keys exhausted after retries."

    def _parse_json_from_response(self, text):
        """Robustly parses JSON from the AI's response."""
        if not text: return None
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        text_to_parse = ""
        if match: text_to_parse = match.group(1)
        else:
            first_brace, last_brace = text.find('{'), text.rfind('}')
            if first_brace != -1 and last_brace > first_brace: text_to_parse = text[first_brace:last_brace+1]
            else: return None
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON. Response snippet: {text_to_parse[:200]}")
        return None

    def _generate_detailed_prompt(self, company_name):
        """Creates an enhanced prompt to find company industry, website, and contacts."""
        return f"""
        **MISSION:** You are an elite AI investigator and data enrichment specialist. Your target is the company "{company_name}" located in or near "{self.args.location}".

        **PRIMARY OBJECTIVE:** Your highest priority is to acquire contact information.

        **EXECUTION PROTOCOL:**
        1.  **IDENTIFY COMPANY DETAILS:** First, determine the company's official website and specific industry (e.g., "Food & Beverage Manufacturing").
        2.  **IDENTIFY CONTACTS:** Use Google Search to find up to 5 contacts. Order them by quality:
            - **Highest Priority:** Direct contacts (C-Level, VP, Director, Head of Department) with direct, professional emails/phones.
            - **Secondary Priority:** If direct contacts are not available, find specific department contacts (e.g., Sales Department).
            - **Fallback:** If no specific contacts are found, find General Inquiry contacts or a main office phone number.
        3.  **GATHER INTEL FOR EACH CONTACT:** For each contact, find their `name`, `title`, `email`, and `phone` number. Use `null` if a piece of information is missing.

        **FINAL REPORTING & QUALITY CONTROL:**
        1.  **INVALIDATION RULE:** Only return `status: "failed"` if the company is non-operational OR you can find absolutely NO contact information of any kind.
        2.  **OUTPUT FORMAT:** Your entire response MUST be a single, raw JSON object. Do not use markdown.

        **JSON STRUCTURE EXAMPLE (SUCCESS):**
        {{
            "status": "success",
            "industry": "Automotive Parts Manufacturing",
            "officialWebsite": "https://www.examplecorp.com",
            "contacts": [
                {{ "name": "Budi Santoso", "title": "Marketing Director", "email": "budi.s@examplecorp.com", "phone": "+6281234567890" }},
                {{ "name": "Sales Department", "title": "General Sales", "email": "sales@examplecorp.com", "phone": null }}
            ]
        }}
        """

    def _process_single_company(self, company_name):
        """Processes one company. Returns True on success, False on failure."""
        prompt = self._generate_detailed_prompt(company_name)
        raw_response, error = self._call_gemini_api(prompt)
        if error:
            logging.error(f"API Error for '{company_name}': {error}")
            return False

        parsed_json = self._parse_json_from_response(raw_response)
        if not parsed_json or parsed_json.get('status') == 'failed':
            reason = parsed_json.get('reason', 'AI returned a failure status or invalid format.')
            logging.warning(f"Lead for '{company_name}' failed. Reason: {reason}")
            return False

        contacts = parsed_json.get('contacts')
        industry = parsed_json.get('industry', 'N/A')
        # --- REVISION: Extract the new website field ---
        website = parsed_json.get('officialWebsite', 'N/A')

        if not isinstance(contacts, list) or not contacts:
            logging.warning(f"Lead for '{company_name}' failed. Reason: Response contained no 'contacts' array.")
            return False

        valid_contacts = [
            c for c in contacts if c.get('email') or c.get('phone')
        ]

        if not valid_contacts:
            logging.warning(f"Lead for '{company_name}' failed. Reason: No contacts with email or phone were found after filtering.")
            return False

        # --- REVISION: Pass the new website variable to the data writer ---
        self.data_manager.write_lead_data(company_name, industry, website, valid_contacts)
        return True

    def _generate_company_candidates(self, count_needed, exclusion_list):
        """Generates a list of new company candidates, avoiding exclusions."""
        logging.info(f"Attempting to generate {count_needed} new company candidates...")
        avoid_list_str = ", ".join(list(exclusion_list)[-200:])
        
        sector_prompt = f"in the '{self.args.sector}' sector" if self.args.sector else "from a variety of promising sectors"

        prompt = (
            f"Act as a Market Research Analyst. Generate a list of {count_needed} high-quality company names "
            f"{sector_prompt}, near '{self.args.location}'. Include promising smaller businesses, not just large corporations. "
            "Provide output as a single line of comma-separated values.\n"
            f"IMPORTANT: Do NOT include any of these company names: {avoid_list_str}"
        )
        
        response_text, error = self._call_gemini_api(prompt)
        if error:
            logging.error(f"Could not generate company list: {error}")
            return []
        
        batch_companies = {name.strip() for name in response_text.split(',') if name.strip()}
        newly_found = list(batch_companies - exclusion_list)
        logging.info(f"Generated {len(newly_found)} unique company candidates.")
        return newly_found

    def run(self):
        """Executes the regenerative lead generation workflow with efficiency improvements."""
        target_company_count = self.args.num_companies
        
        searched_companies = self.data_manager.get_existing_company_names()
        successful_company_count = 0
        
        consecutive_generation_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3

        while successful_company_count < target_company_count:
            needed_count = target_company_count - successful_company_count
            logging.info(f"Goal: {target_company_count} successful companies. Current: {successful_company_count}. Need {needed_count} more.")

            companies_to_process = self._generate_company_candidates(needed_count, searched_companies)
            
            if not companies_to_process:
                consecutive_generation_failures += 1
                logging.warning(f"Could not generate new candidates. Failure {consecutive_generation_failures}/{MAX_CONsecutive_FAILURES}.")
                if consecutive_generation_failures >= MAX_CONsecutive_FAILURES:
                    logging.error("Exceeded max consecutive failures to generate new companies. Halting.")
                    break
                continue
            
            consecutive_generation_failures = 0
            searched_companies.update(companies_to_process)
            
            batch_success_count = 0
            with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix='LeadGen') as executor:
                future_to_company = {executor.submit(self._process_single_company, c): c for c in companies_to_process}
                for future in as_completed(future_to_company):
                    company_name = future_to_company[future]
                    try:
                        if future.result():
                            batch_success_count += 1
                            logging.info(f"Successfully processed lead for: {company_name}")
                    except Exception as e:
                        logging.error(f"Task for {company_name} generated an exception: {e}", exc_info=True)
            
            successful_company_count += batch_success_count
        
        logging.info(f"Workflow finished. Successfully processed {successful_company_count} / {target_company_count} targeted companies.")


def main():
    parser = argparse.ArgumentParser(description="Generates high-quality leads and saves them to a local CSV file.")
    parser.add_argument('--location', type=str, required=True, help="Target location for leads.")
    parser.add_argument('--sector', type=str, required=False, help="Optional: Specify an industry sector. If omitted, searches across random sectors.")
    parser.add_argument('--num_companies', type=int, default=10, help="Total number of SUCCESSFUL companies to generate leads for.")
    parser.add_argument('--workers', type=int, default=5, help="Number of parallel threads for processing.")
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
