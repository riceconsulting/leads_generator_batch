# File: validate_leads.py
# --- REVISED WITH INTELLIGENT CORRECTION LOGIC AND ANTI-HALLUCINATION PROMPTS ---

import os
import time
import argparse
import logging
import json
import csv
import re
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming 'google.generativeai' is installed from requirements.txt
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
load_dotenv()

# --- REUSABLE CLASSES (Copied from leads_generator.py for standalone use) ---

class Config:
    """Loads and validates all necessary environment variables."""
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

# --- CORE VALIDATION LOGIC ---

class LeadValidator:
    """Orchestrates the validation and correction of leads from the CSV file."""
    def __init__(self, config, args):
        self.key_manager = GeminiAPIKeyManager(config.gemini_api_keys)
        self.args = args
        self.output_folder = Path("lead_results")
        self.csv_file = self.output_folder / "lead_results.csv"

    def _call_gemini_api(self, prompt):
        """Calls Gemini API with retries and the Google Search tool."""
        max_retries_per_key = 5
        while True:
            api_key = self.key_manager.get_key()
            if not api_key: return None, "All API keys are exhausted."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash-latest',
                tools=['google_search']
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

    def _create_validation_prompt(self, company_batch):
        """Creates a prompt to validate and correct a batch of companies."""
        company_list_str = json.dumps(company_batch, indent=2)
        return f"""
        **MISSION:** You are a meticulous AI Data Validation Specialist. Your task is to validate and, if necessary, correct a list of business leads. You MUST base your findings on verifiable public sources found via Google Search. Do NOT invent information.

        **LEADS TO VALIDATE:**
        {company_list_str}

        **EXECUTION PROTOCOL:**
        For each company, perform these checks:
        1.  **Existence & Quality:** Verify the company is a real, operational business suitable for B2B engagement.
        2.  **Name & Website Verification:**
            - Check if the provided `name` is the official, correct name.
            - Check if the `website` is the official corporate site, not a directory or social media.
        3.  **Determine Status:**
            - If name and website are correct, status is **VALID**.
            - If the company exists but the name or website is incorrect, status is **CORRECTED**. Provide the accurate `correctedName` and `correctedWebsite`.
            - If the company does not exist, is not a real business, or the website is an irrelevant link (like a directory), status is **INVALID**.

        **FINAL REPORTING & QUALITY CONTROL:**
        - Your response MUST be a single, raw JSON object. Do not use markdown.
        - The JSON object should have the original company names as keys.
        - For each company, provide a nested object with the following keys: `status` ("VALID", "INVALID", or "CORRECTED"), `correctedName`, `correctedWebsite`, and `reason`.

        **JSON STRUCTURE EXAMPLE:**
        {{
          "PT Example Corp": {{ "status": "VALID", "correctedName": "PT Example Corp", "correctedWebsite": "https://examplecorp.com", "reason": "Verified." }},
          "Example Corp Indonesia": {{ "status": "CORRECTED", "correctedName": "PT Example Corporation Tbk", "correctedWebsite": "https://example-corp-indonesia.com", "reason": "Corrected official company name." }},
          "Surabaya Directory Inc": {{ "status": "INVALID", "correctedName": null, "correctedWebsite": null, "reason": "Website is a generic business directory, not an official company site." }}
        }}
        """

    def _validate_batch(self, company_batch_data):
        """Processes one batch of companies against the Gemini API."""
        prompt = self._create_validation_prompt(company_batch_data)
        raw_response, error = self._call_gemini_api(prompt)
        if error:
            logging.error(f"API Error for batch: {error}")
            return {item['name']: {'status': 'INVALID', 'reason': 'API Error'} for item in company_batch_data}
        
        parsed_json = self._parse_json_from_response(raw_response)
        if not parsed_json:
            return {item['name']: {'status': 'INVALID', 'reason': 'Failed to parse AI response'} for item in company_batch_data}
        return parsed_json

    def run(self):
        """Executes the entire lead validation, correction, and cleaning workflow."""
        if not self.csv_file.exists():
            logging.error(f"'{self.csv_file}' not found. Nothing to validate.")
            return

        all_rows = []
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        
        if not all_rows:
            logging.info("CSV is empty. Nothing to validate.")
            return

        companies_grouped = defaultdict(list)
        for row in all_rows:
            companies_grouped[row['CompanyName']].append(row)
        
        unique_companies = list(companies_grouped.keys())
        logging.info(f"Found {len(unique_companies)} unique companies to validate from {len(all_rows)} total rows.")
        
        batch_size = 10
        company_batches = [unique_companies[i:i + batch_size] for i in range(0, len(unique_companies), batch_size)]
        
        final_rows_to_keep = []
        
        with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix='Validator') as executor:
            future_to_batch = {}
            for batch in company_batches:
                batch_data_for_prompt = [{'name': name, 'website': companies_grouped[name][0].get('CompanyWebsite', '')} for name in batch]
                future = executor.submit(self._validate_batch, batch_data_for_prompt)
                future_to_batch[future] = batch

            for future in as_completed(future_to_batch):
                batch_results = future.result()
                for original_name, result_data in batch_results.items():
                    status = result_data.get('status', 'INVALID')
                    
                    if status == 'VALID':
                        logging.info(f"VALIDATING Company: '{original_name}'")
                        final_rows_to_keep.extend(companies_grouped[original_name])
                    
                    elif status == 'CORRECTED':
                        corrected_name = result_data.get('correctedName', original_name)
                        corrected_website = result_data.get('correctedWebsite')
                        reason = result_data.get('reason', 'No reason provided.')
                        logging.warning(f"CORRECTING Company: '{original_name}' -> '{corrected_name}'. Reason: {reason}")
                        
                        original_rows = companies_grouped[original_name]
                        for row in original_rows:
                            row['CompanyName'] = corrected_name
                            row['CompanyWebsite'] = corrected_website
                        final_rows_to_keep.extend(original_rows)

                    else: # Status is INVALID
                        reason = result_data.get('reason', 'No reason provided.')
                        logging.warning(f"REMOVING Company: '{original_name}'. Reason: {reason}")
        
        if not final_rows_to_keep:
            logging.warning("Validation resulted in 0 valid leads. The CSV will be cleared.")
        
        header = ['CompanyName', 'Industry', 'CompanyWebsite', 'ContactName', 'ContactTitle', 'Email', 'PhoneNumber', 'Timestamp']
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(final_rows_to_keep)

        logging.info(f"Validation complete. Kept {len(final_rows_to_keep)} rows for {len(set(row['CompanyName'] for row in final_rows_to_keep))} companies. "
                     f"Removed/corrected data from {len(all_rows) - len(final_rows_to_keep)} original rows.")


def main():
    parser = argparse.ArgumentParser(description="Validates and cleans the leads in lead_results.csv.")
    parser.add_argument('--workers', type=int, default=5, help="Number of parallel threads for processing.")
    args = parser.parse_args()

    logging.info("Starting Lead Validation and Correction Script...")
    try:
        config = Config()
        validator = LeadValidator(config, args)
        validator.run()
    except ValueError as e:
        logging.error(f"Configuration Error: {e}")
    except Exception as e:
        logging.error(f"A critical, unhandled error occurred: {e}", exc_info=True)
    logging.info("Lead Validation and Correction Script finished.")

if __name__ == "__main__":
    main()
