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
import csv
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
#
#   PURPOSE: Build system prompt + listing data for LLM classification
#   INPUT:   clean signal dict from extract_signal_fields()
#   OUTPUT:  full prompt string (system instructions + formatted listing fields)

def build_prompt(listing: dict) -> str:
    system = """You are a property classification assistant for a UK property acquisition and development company called Harkalm Group.

The company acquires and develops properties across three sectors only. Your job is to classify each listing into one of four categories based on the available listing data.

CATEGORIES:

NURSERY: Children's day care or early years education. Strong signals include:
- Explicit mention of children's nursery, day care, or early years provision
- D1, E(f), or F1 planning consent for childcare use
- Age ranges for young children (0-5, birth to 5)
- Ofsted registration or funded childcare places
- Nursery-specific fixtures: low-level sinks, enclosed outdoor play areas, soft play
- Vacant commercial buildings (former banks, offices) in residential catchments
  with conversion potential may qualify even without current nursery use

SEN SCHOOL: Special educational needs provision. Strong signals include:
- Explicit mention of SEN, SEMH, alternative provision, or specialist school
- F1 (formerly D1) planning consent for educational institution
- Sensory rooms, therapy suites, or specialist teaching facilities
- Local authority placements or pupil referrals mentioned
- School-age children (5-18) in specialist or therapeutic settings

FOOD STORE: Convenience or grocery retail specifically. Strong signals include:
- Named grocery retailer (Co-op, Costcutter, Tesco, Sainsbury's, Aldi, Lidl etc.)
- A1 or E use class with explicit food grocery retail context
- Chiller cabinets, shelving runs, stockroom, rear servicing yard
- Supermarket or convenience store explicitly named
- Property size broadly 2,500 to 30,000 sq ft
- IMPORTANT: Coffee shops, restaurants, bars, takeaways, and pubs are NOT
  food stores for this company — classify these as None

NONE: Use when:
- The listing is residential, land, industrial, pub, or mixed-use with no clear
  sector fit
- Agent copy hedges with phrases like "suitable for alternative uses",
  "variety of uses", or "may suit" without a prior sector-specific operator,
  confirmed use class, or sector-specific fixtures to anchor the classification
- There is insufficient signal to confidently assign any of the three sectors
- The property is legally or structurally suited to a use outside the three
  sectors above (e.g. healthcare, office, retail restaurant)

CONFIDENCE RULES:
High   — Sector-specific evidence is explicit and unambiguous: named operator,
          confirmed planning consent for that use, sector fixtures present,
          or property currently/recently operating in that sector
Medium — Prior use matches a sector but the listing hedges on future use,
          or evidence is present but indirect (e.g. former food store now
          vacant with no stated future intent, or conversion potential
          without confirmed planning)
Low    — Minimal signal, genuine ambiguity, agent copy that could fit
          multiple uses, or a single weak indicator with no corroboration

IMPORTANT GUIDANCE ON AMBIGUOUS CASES:
- "Suitable for a variety of alternative uses" with no sector anchor = None, Low
- Former food store, vacant, no stated intent = Food Store, Medium
- Former nursery with fixtures intact = Nursery, High
- Healthcare or medical professional use = None (not one of the three sectors)
- Pub or restaurant conversion potential only = None
- Do not force a sector label when the evidence is genuinely ambiguous —
  None with Low confidence is a valid and expected output for some listings

OUTPUT FORMAT:
Respond with valid JSON only. No markdown formatting, no code blocks, no preamble,
no explanation outside the JSON object. Your entire response must be parseable JSON.

{
  "classification": "Nursery | SEN School | Food Store | None",
  "confidence": "High | Medium | Low",
  "reasoning": "One or two sentences citing the specific evidence that drove this decision. Reference the actual listing data, not generic statements."
}"""

    listing_text = "\n".join([
        f"summary: {listing['summary'] or 'not provided'}",
        f"detailedDescription: {listing['detailedDescription'] or 'not provided'}",
        f"keyFeatures: {', '.join(listing['keyFeatures']) if listing['keyFeatures'] else 'none'}",
        f"propertySubType: {listing['propertySubType'] or 'not provided'}",
        f"useClass: {listing['useClass'] or 'not provided'}",
        f"tenureType: {listing['tenureType'] or 'not provided'}",
        f"pageTitle: {listing['pageTitle'] or 'not provided'}",
    ])

    return f"{system}\n\nLISTING DATA:\n{listing_text}"

# ─────────────────────────────────────────────────────────────────────────────

# call_llm(prompt: str) -> str
#
#   PURPOSE: Send prompt to LLM API, return raw response text
#   INPUT:   prompt string
#   OUTPUT:  raw response string
#   HANDLES:
#     - API errors -> raise with informative message
#     - timeout -> retry once, then raise

def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API call failed: {e}")

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

def parse_llm_response(response_text: str) -> dict:
    valid_categories = {"Nursery", "SEN School", "Food Store", "None"}
    valid_confidence = {"High", "Medium", "Low"}

    try:
        # strip markdown code blocks if model wraps response
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        classification = parsed.get("classification", "None")
        confidence = parsed.get("confidence", "Low")
        reasoning = parsed.get("reasoning", "No reasoning provided")

        if classification is None or (
            isinstance(classification, float) and pd.isna(classification)
        ):
            classification = "None"
        if classification not in valid_categories:
            classification = "None"
        if confidence is None or (
            isinstance(confidence, float) and pd.isna(confidence)
        ):
            confidence = "Low"
        if confidence not in valid_confidence:
            confidence = "Low"
        if reasoning is None or (
            isinstance(reasoning, float) and pd.isna(reasoning)
        ):
            reasoning = "No reasoning provided"

        return {
            "classification": str(classification),
            "confidence": str(confidence),
            "reasoning": str(reasoning),
        }

    except (json.JSONDecodeError, KeyError, IndexError):
        return {
            "classification": "None",
            "confidence": "Low",
            "reasoning": "Failed to parse LLM response",
        }

# ─────────────────────────────────────────────────────────────────────────────

# classify_listing(listing: dict) -> dict
#
#   PURPOSE: Orchestrate single listing classification end to end
#   INPUT:   clean signal dict
#   OUTPUT:  {id, classification, confidence, reasoning}
#   CALLS:   build_prompt() -> call_llm() -> parse_llm_response()

def classify_listing(listing: dict) -> dict:
    prompt = build_prompt(listing)
    raw = call_llm(prompt)
    parsed = parse_llm_response(raw)
    return {"id": listing["id"], **parsed}

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

def write_results(results: list, original_df: pd.DataFrame, output_path: str):
    results_df = pd.DataFrame(results)

    # normalise id column in original for merge
    original_df["id"] = original_df["id"].apply(normalise_id)

    merged = original_df.merge(results_df, on="id", how="left")
    merged["classification"] = merged["classification"].fillna("None")
    merged["confidence"] = merged["confidence"].fillna("Low")
    merged["reasoning"] = merged["reasoning"].fillna("Insufficient data to classify")
    merged.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Results written to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────

# main()
#
#   PURPOSE: Orchestrate full pipeline
#   FLOW:
#     1. load_csv(INPUT_PATH) -> raw_df
#     2. for each row in raw_df:
#            listing = extract_signal_fields(row)
#            result  = classify_listing(listing)
#            append result to results list
#     3. write_results(results, raw_df, OUTPUT_PATH)
#     4. print summary: N classified, breakdown by category

def main():
    df = load_csv(INPUT_PATH)
    listings = [extract_signal_fields(row) for _, row in df.iterrows()]

    results = []
    for listing in listings:
        print(f"Classifying {listing['id']}...")
        result = classify_listing(listing)
        results.append(result)
        print(f"  -> {result['classification']} ({result['confidence']})")

    write_results(results, df, OUTPUT_PATH)

    # print summary from in-memory results
    from collections import Counter
    counts = Counter(r["classification"] for r in results)
    print("\nSummary:")
    for category, count in counts.items():
        print(f"  {category}: {count}")

    # verify CSV round-trip (pandas treats bare "None" as NA unless disabled)
    verify_df = pd.read_csv(OUTPUT_PATH, keep_default_na=False)
    csv_counts = verify_df["classification"].value_counts()
    print("\nVerified CSV value_counts:")
    for category, count in csv_counts.items():
        print(f"  {category}: {count}")
    print(f"  Total: {len(verify_df)}")


# ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    main()
