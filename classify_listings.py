# classify_listings.py
# Property Listing Classifier
#
# OVERVIEW:
# Pipeline that reads messy property listing data, extracts signal-rich fields,
# classifies each listing into one of four categories using an LLM, and writes
# results back alongside the original data.
#
# INPUTS:  listings.csv (23 property listings, ~70 columns, mostly sparse)
# OUTPUTS: results.csv  (original data + classification, confidence, reasoning)
#
# CATEGORIES: Nursery | SEN School | Food Store | None
# CONFIDENCE: High | Medium | Low
#
# ─────────────────────────────────────────────────────────────────────────────

import ast
import pandas as pd

# --- CONSTANTS ---
# INPUT_PATH  = "listings.csv"
# OUTPUT_PATH = "results.csv"
# MODEL       = "gpt-4o-mini"
# SIGNAL_FIELDS = [
#     "id", "summary", "detailedDescription", "keyFeatures",
#     "propertySubType", "useClass", "tenureType", "pageTitle"
# ]

INPUT_PATH = "listings.csv"
OUTPUT_PATH = "results.csv"
MODEL = "gpt-4o-mini"
SIGNAL_FIELDS = [
    "id", "summary", "detailedDescription", "keyFeatures",
    "propertySubType", "useClass", "tenureType", "pageTitle"
]

# ─────────────────────────────────────────────────────────────────────────────

# load_csv(filepath: str) -> pd.DataFrame
#
#   PURPOSE: Load raw listings CSV into a dataframe
#   INPUT:   filepath string
#   OUTPUT:  raw dataframe, all 70 columns, all 23 rows
#   HANDLES: file not found, encoding errors

def load_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, dtype=str)
        print(f"Loaded {len(df)} listings from {filepath}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find listings file: {filepath}")

# ─────────────────────────────────────────────────────────────────────────────

# extract_signal_fields(row: pd.Series) -> dict
#
#   PURPOSE: Extract only the fields with classification signal from a raw row.
#            Most columns are empty for most rows - this filters the noise.
#   INPUT:   single dataframe row
#   OUTPUT:  clean dict, e.g.:
#            {
#              "id": "HK-F001",
#              "summary": "FOR SALE - Freehold former Co-op...",
#              "detailedDescription": "Freehold retail unit comprising...",
#              "keyFeatures": ["A1 Use Class", "Former Co-op", "1,850 sq ft"],
#              "propertySubType": "retail",
#              "useClass": "A1",
#              "tenureType": "FREEHOLD",
#              "pageTitle": "..."
#            }
#   HANDLES:
#     - missing/NaN fields -> omit from dict or use empty string
#     - keyFeatures is a stringified Python list -> ast.literal_eval()
#       with try/except fallback to raw string if malformed
#     - numeric IDs (e.g. 89924760.0) -> cast to string, strip ".0"

def parse_key_features(raw: str) -> list[str]:
    # keyFeatures is a stringified Python list
    # try ast.literal_eval first, fall back to raw string
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [raw] if raw else []


def normalise_id(raw_id: str) -> str:
    # numeric IDs come through as "89924760.0" — strip the .0
    try:
        return str(int(float(raw_id)))
    except (ValueError, TypeError):
        return str(raw_id).strip()


def extract_signal_fields(row: pd.Series) -> dict:
    def get(field: str) -> str:
        val = row.get(field, "")
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            return ""
        return str(val).strip()

    return {
        "id": normalise_id(get("id")),
        "summary": get("summary"),
        "detailedDescription": get("detailedDescription"),
        "keyFeatures": parse_key_features(get("keyFeatures")),
        "propertySubType": get("propertySubType"),
        "useClass": get("useClass"),
        "tenureType": get("tenureType"),
        "pageTitle": get("pageTitle"),
    }

# ─────────────────────────────────────────────────────────────────────────────

# build_prompt(listing: dict) -> str
#   TO BE IMPLEMENTED - see future commit

# ─────────────────────────────────────────────────────────────────────────────

# call_llm(prompt: str, client) -> str
#
#   PURPOSE: Send prompt to LLM API, return raw response text
#   INPUT:   prompt string, initialised API client
#   OUTPUT:  raw response string
#   HANDLES:
#     - API errors -> raise with informative message
#     - timeout -> retry once, then raise

# ─────────────────────────────────────────────────────────────────────────────

# parse_llm_response(response_text: str) -> dict
#
#   PURPOSE: Parse raw LLM response into structured classification dict
#   INPUT:   raw response string (should be JSON)
#   OUTPUT:  {classification, confidence, reasoning}
#   HANDLES:
#     - JSON parse errors -> return
#       {classification: "None", confidence: "Low", reasoning: "Parse error"}
#     - Missing keys -> fill with safe defaults
#     - Invalid category values -> default to "None"
#     - Invalid confidence values -> default to "Low"

# ─────────────────────────────────────────────────────────────────────────────

# classify_listing(listing: dict, client) -> dict
#
#   PURPOSE: Orchestrate single listing classification end to end
#   INPUT:   clean signal dict, API client
#   OUTPUT:  {id, classification, confidence, reasoning}
#   CALLS:   build_prompt() -> call_llm() -> parse_llm_response()

# ─────────────────────────────────────────────────────────────────────────────

# write_results(results: list[dict], original_df: pd.DataFrame, output_path: str)
#
#   PURPOSE: Merge classification results with original data, write to CSV
#   INPUT:   list of {id, classification, confidence, reasoning}
#            original dataframe (all 70 columns)
#            output path string
#   OUTPUT:  results.csv
#            - all original columns preserved
#            - three columns appended on right: classification, confidence, reasoning
#            - merged on id column
#   HANDLES: id type mismatch (string vs float) during merge

# ─────────────────────────────────────────────────────────────────────────────

# main()
#
#   PURPOSE: Orchestrate full pipeline
#   FLOW:
#     1. load_csv(INPUT_PATH) -> raw_df
#     2. for each row in raw_df:
#            listing = extract_signal_fields(row)
#            result  = classify_listing(listing, client)
#            append result to results list
#     3. write_results(results, raw_df, OUTPUT_PATH)
#     4. print summary: N classified, breakdown by category

def main():
    df = load_csv(INPUT_PATH)
    listings = []
    for _, row in df.iterrows():
        listing = extract_signal_fields(row)
        listings.append(listing)
        print(f"ID: {listing['id']} | summary: {listing['summary'][:80]}")

    print(f"\nExtracted {len(listings)} listings")
    return listings

# ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    main()
