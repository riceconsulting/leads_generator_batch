# File: leads_generator.py
# --- REVISED WITH ALL CORRECTIONS AND MODEL UPDATE ---

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
        """Initializes the CSV with the new headers if it doesn't exist."""
        with self.csv_lock:
            if not self.output_file.exists():
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'CompanyName', 'Status', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber',
                        'AlternativeEmail', 'AlternativePhoneNumber', 'LinkedIn', 'Timestamp'
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
        """Writes a single lead to the CSV, mapping the new data structure."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        status = lead_data.get('status', 'Failed')
        get = lambda key: lead_data.get(key) or ""

        with self.csv_lock:
            try:
                with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        company_name, status, get('contact_name'), get('contact_title'), get('email'),
                        get('phone_number'), get('alternative_email'), get('alternative_phone_number'),
                        get('linkedin_url'), timestamp
                    ])
            except Exception as e:
                logging.error(f"Failed to write to CSV for '{company_name}': {e}")


class LeadGenerationOrchestrator:
    """Manages the entire lead generation process with advanced prompting and parsing."""
    def __init__(self, config, args):
        self.key_manager = GeminiAPIKeyManager(config.gemini_api_keys)
        self.data_manager = CSVDataManager()
        self.args = args

    def _call_gemini_api(self, prompt):
        """Calls Gemini API with increased retries and the Google Search tool."""
        max_retries_per_key = 5
        while True:
            api_key = self.key_manager.get_key()
            if not api_key: return None, "All API keys are exhausted."
            genai.configure(api_key=api_key)
            
            # --- FIX: Using the requested 'flash' model while keeping search tool for high quality ---
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
        """Robustly parses JSON from the AI's response, handling markdown fences."""
        if not text:
            return None
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON from detected markdown block.")
        try:
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace > first_brace:
                return json.loads(text[first_brace:last_brace+1])
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from raw text fallback. Response: {text[:200]}")
        return None

    def _generate_detailed_prompt(self, company_name):
        """Creates a highly detailed, robust prompt focused on contact acquisition."""
        return f"""
        **MISSION:** You are an elite AI investigator. Your target is the company "{company_name}" in the region of "{self.args.location}". Your investigation must be thorough and swift.

        **PRIMARY OBJECTIVE (NON-NEGOTIABLE):** Your absolute highest priority is to acquire high-value contact points. Success is defined as finding AT LEAST ONE of the following:
        1. A direct email address for a key decision-maker (e.g., firstname.lastname@company.com).
        2. A direct mobile phone number for a key staff member.

        If you cannot acquire at least one of these, the lead is considered INVALID. General "info@" emails or generic company hotlines are LOW-VALUE and should only be used as alternatives if direct contacts are found.

        **EXECUTION PROTOCOL:**
        1.  **VIABILITY:** Use Google Search to confirm "{company_name}" is operational. If it is permanently closed or un-findable, the lead is INVALID.
        2.  **GATHER INTEL:** Find the following information:
            *   `contactPerson`: name and title of a relevant decision-maker (e.g., Owner, Director, Head of Sales, Head of Marketing).
            *   `contactEmail`: An array of strings. Prioritize direct emails. Include high-quality general emails (e.g., sales@) as secondary options.
            *   `contactPhone`: An array of strings. Prioritize direct mobile numbers. Include high-quality company phone numbers as secondary options.
            *   `linkedinUrl`: The LinkedIn URL of the contact person, if available.
        3.  **EXCLUDE LOW-VALUE CONTACTS:** Ignore and do not include `jobs@`, `careers@`, `support@`, `help@`, etc., unless no other contacts can be found.

        **FINAL REPORTING & QUALITY CONTROL:**
        1.  **COMPILE JSON:** Assemble all gathered data into the specified JSON structure. All fields must be present.
        2.  **INVALIDATION RULE:** If the lead is INVALID (non-operational or fails the PRIMARY OBJECTIVE), your entire response MUST be ONLY: `{{ "status": "failed", "reason": "Could not find direct contact info." }}`.
        3.  **OUTPUT FORMAT:** Your entire response MUST be a single, raw JSON object. Do not use markdown fences (```json).

        **JSON STRUCTURE EXAMPLE (SUCCESS):**
        {{
            "status": "success",
            "contactPerson": {{ "name": "Budi Santoso", "title": "Marketing Director" }},
            "contactDetails": {{
                "emails": ["budi.s@examplecorp.com", "sales.surabaya@examplecorp.com"],
                "phones": ["+6281234567890", "+62315551234"],
                "linkedinUrl": "https://linkedin.com/in/budisantoso"
            }}
        }}
        """

    def _process_single_company(self, company_name):
        """Processes one company, applying the new robust logic."""
        prompt = self._generate_detailed_prompt(company_name)
        raw_response, error = self._call_gemini_api(prompt)
        
        if error:
            lead_data = {'status': 'Failed', 'reason': f"API Error: {error}"}
            self.data_manager.write_lead_data(company_name, lead_data)
            return company_name

        parsed_json = self._parse_json_from_response(raw_response)

        if not parsed_json or parsed_json.get('status') == 'failed':
            reason = parsed_json.get('reason', 'Invalid or failed response from AI.')
            logging.warning(f"Lead for '{company_name}' marked as invalid. Reason: {reason}")
            lead_data = {'status': 'Failed'}
            self.data_manager.write_lead_data(company_name, lead_data)
            return company_name

        contact_person = parsed_json.get('contactPerson', {})
        contact_details = parsed_json.get('contactDetails', {})
        emails = contact_details.get('emails', [])
        phones = contact_details.get('phones', [])

        # --- FIX: Correctly map list items to individual CSV columns ---
        lead_data = {
            'status': 'Processed',
            'contact_name': contact_person.get('name'),
            'contact_title': contact_person.get('title'),
            'email': emails if emails else None,
            'alternative_email': emails if len(emails) > 1 else None,
            'phone_number': phones if phones else None,
            'alternative_phone_number': phones if len(phones) > 1 else None,
            'linkedin_url': contact_details.get('linkedinUrl')
        }
        self.data_manager.write_lead_data(company_name, lead_data)
        return company_name

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


def main():
    parser = argparse.ArgumentParser(description="Generates leads and saves them to a local CSV file.")
    # --- FIX: Corrected 'add-argument' typo ---
    parser.add_argument('--location', type=str, required=True, help="Target location for leads.")
    parser.add_argument('--sector', type=str, required=True, help="Industry sector for leads.")
    parser.add_argument('--num_companies', type=int, default=10, help="Total number of new companies to generate.")
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
