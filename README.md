# Harkalm Property Classifier

Classifies property listings into **Nursery**, **SEN School**, **Food Store**, or **None** using an LLM API.

## Setup & Run

```bash
pip install -r requirements.txt
```

Add your OpenAI API key to a `.env` file: `OPENAI_API_KEY=sk-...`

```bash
python classify_listings.py
```

Input: `listings.csv` → Output: `results.csv`

## Approach

Most of the ~70 columns are empty for any given row. I selected the seven fields that consistently carry classification signal: `summary`, `detailedDescription`, `keyFeatures`, `propertySubType`, `useClass`, `tenureType`, and `pageTitle`. The `keyFeatures` field is a stringified Python list and is parsed with `ast.literal_eval()` with a graceful fallback.

The classification prompt is built around Harkalm's actual acquisition criteria. Confidence is High when sector evidence is explicit (named operator, confirmed planning consent, sector-specific fixtures), Medium when prior use matches but future intent hedges, and Low when agent copy is genuinely ambiguous. Listings using "suitable for a variety of alternative uses" without a sector anchor are classified as None — the ambiguity is the signal. The former Costcutter is Food Store at Medium confidence (prior use is meaningful; hedged future intent reduces certainty). The healthcare facility is None — medical use falls outside Harkalm's three sectors. Temperature is set to 0 for deterministic output.

## What I'd Improve With More Time

For low-signal or ambiguous listings, I'd add an agentic enrichment step that uses the property's geolocation and address to query planning portals, Google Maps, and council databases — pulling operator presence and planning history to resolve borderline cases. I'd also add parallel API calls, a second-pass validation for Medium/Low results, unit tests for the parsing functions, and per-run cost logging.
