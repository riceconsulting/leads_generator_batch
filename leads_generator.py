# File: leads_generator.py
# --- REVISED WITH REGENERATIVE FAILURE LOOP, NEW CSV FORMAT, AND ENHANCED PROMPTS ---

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
                # --- FIX: Revised headers as requested ---
                writer.writerow([
                    'CompanyName', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber',
                    'AlternativeContactName', 'AlternativeContactTitle', 'AlternativeEmail', 'AlternativePhoneNumber',
                    'Timestamp'
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
        """Writes a single successful lead to the CSV."""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        get = lambda key: lead_data.get(key) or ""

        with self.csv_lock:
            try:
                with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # --- FIX: Writing data in the new column order ---
                    writer.writerow([
                        company_name, get('contact_name'), get('contact_title'), get('email'), get('phone_number'),
                        get('alt_contact_name'), get('alt_contact_title'), get('alt_email'), get('alt_phone_number'),
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
        if not text:
            return None
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            text_to_parse = match.group(1)
        else:
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace > first_brace:
                text_to_parse = text[first_brace:last_brace+1]
            else:
                return None
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON. Response snippet: {text_to_parse[:200]}")
        return None

    def _generate_detailed_prompt(self, company_name):
        """Creates an enhanced, robust prompt for finding multiple contacts."""
        return f"""
        **MISSION:** You are an elite AI investigator and data enrichment specialist. Your target is the company "{company_name}" within the "{self.args.sector}" sector in or near "{self.args.location}".

        **PRIMARY OBJECTIVE (NON-NEGOTIABLE):** Your highest priority is to acquire direct contact information for key decision-makers. Success is defined as finding AT LEAST ONE direct email OR direct phone number.

        **EXECUTION PROTOCOL:**
        1.  **IDENTIFY CONTACTS:** Use Google Search to find up to two key decision-makers. Prioritize roles like C-Level (CEO, CTO), VP, Director, or Head of Department (Sales, Marketing, Operations).
        2.  **GATHER INTEL FOR EACH CONTACT:** For each person you identify, find their `name`, `title`, direct `email`, and direct `phone` number.
        3.  **EXCLUDE LOW-VALUE DATA:** Do NOT include generic emails (`info@`, `support@`, `jobs@`) or general company hotlines unless NO direct information can be found.

        **FINAL REPORTING & QUALITY CONTROL:**
        1.  **INVALIDATION RULE:** If the company is not operational OR you fail the PRIMARY OBJECTIVE (cannot find a single direct email or phone number), your entire response MUST be ONLY: `{{ "status": "failed", "reason": "Could not find any direct contact information." }}`.
        2.  **OUTPUT FORMAT:** Your entire response MUST be a single, raw JSON object. Do not use markdown.

        **JSON STRUCTURE EXAMPLE (SUCCESS):**
        {{
            "status": "success",
            "contacts": [
                {{
                    "name": "Budi Santoso",
                    "title": "Marketing Director",
                    "email": "budi.s@examplecorp.com",
                    "phone": "+6281234567890"
                }},
                {{
                    "name": "Ani Wijaya",
                    "title": "Head of Sales",
                    "email": "ani.wijaya@examplecorp.com",
                    "phone": null
                }}
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

        if not parsed_json or parsed_json.get('status') == 'failed' or not parsed_json.get('contacts'):
            reason = parsed_json.get('reason', 'Invalid or failed response from AI.')
            logging.warning(f"Lead for '{company_name}' failed. Reason: {reason}")
            return False

        contacts = parsed_json.get('contacts', [])
        if not isinstance(contacts, list) or len(contacts) == 0:
            logging.warning(f"Lead for '{company_name}' failed. Reason: No contacts found in the response.")
            return False

        # Map the successful JSON response to our flat CSV structure
        primary_contact = contacts[0]
        secondary_contact = contacts[1] if len(contacts) > 1 else {}
        
        lead_data = {
            'contact_name': primary_contact.get('name'),
            'contact_title': primary_contact.get('title'),
            'email': primary_contact.get('email'),
            'phone_number': primary_contact.get('phone'),
            'alt_contact_name': secondary_contact.get('name'),
            'alt_contact_title': secondary_contact.get('title'),
            'alt_email': secondary_contact.get('email'),
            'alt_phone_number': secondary_contact.get('phone'),
        }
        self.data_manager.write_lead_data(company_name, lead_data)
        return True

    def _generate_company_candidates(self, count_needed, exclusion_list):
        """Generates a list of new company candidates, avoiding exclusions."""
        logging.info(f"Attempting to generate {count_needed} new company candidates...")
        avoid_list_str = ", ".join(list(exclusion_list)[-200:])
        
        prompt = (
            f"Act as a Market Research Analyst. Generate a list of {count_needed} high-quality company names "
            f"in the '{self.args.sector}' sector, near '{self.args.location}'. "
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
        """Executes the regenerative lead generation workflow."""
        target_lead_count = self.args.num_companies
        successful_leads_count = 0
        searched_companies = self.data_manager.get_existing_company_names()
        max_attempts = 3 # Safety break to prevent infinite loops
        current_attempt = 0

        while successful_leads_count < target_lead_count and current_attempt < max_attempts:
            needed_count = target_lead_count - successful_leads_count
            logging.info(f"Goal: {target_lead_count} leads. Current: {successful_leads_count}. Need {needed_count} more.")

            # Phase 1: Generate new, unique company candidates
            companies_to_process = self._generate_company_candidates(needed_count, searched_companies)
            
            if not companies_to_process:
                logging.warning("Could not generate any new company candidates in this attempt.")
                current_attempt += 1
                continue

            searched_companies.update(companies_to_process)
            
            # Phase 2: Process companies in parallel
            batch_success_count = 0
            with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix='LeadGen') as executor:
                future_to_company = {executor.submit(self._process_single_company, c): c for c in companies_to_process}
                for future in as_completed(future_to_company):
                    company_name = future_to_company[future]
                    try:
                        is_success = future.result()
                        if is_success:
                            batch_success_count += 1
                            logging.info(f"Successfully processed lead for: {company_name}")
                        else:
                            logging.info(f"Failed to process lead for: {company_name}")
                    except Exception as e:
                        logging.error(f"Task for {company_name} generated an exception: {e}", exc_info=True)

            if batch_success_count == 0:
                current_attempt += 1
            else:
                current_attempt = 0 # Reset safety break if we are making progress

            successful_leads_count += batch_success_count

        logging.info(f"Workflow finished. Successfully generated {successful_leads_count} / {target_lead_count} targeted leads.")


def main():
    parser = argparse.ArgumentParser(description="Generates high-quality leads and saves them to a local CSV file.")
    parser.add_argument('--location', type=str, required=True, help="Target location for leads.")
    parser.add_argument('--sector', type=str, required=True, help="Industry sector for leads.")
    parser.add_argument('--num_companies', type=int, default=10, help="Total number of SUCCESSFUL leads to generate.")
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
