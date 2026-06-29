# PropertyClassifier

Classifies UK property listings into Harkalm Group acquisition sectors (Nursery, SEN School, Food Store) using an LLM.

## Usage

```powershell
pip install -r requirements.txt
python classify_listings.py
```

Reads `listings.csv`, writes `results.csv` with classification, confidence, and reasoning columns appended.

## Known limitations

**LLM non-determinism:** Classifications may vary slightly between runs even with `temperature=0`. For example, a former food store with fixtures intact may be rated High in one run and Medium in another ("vacant, no stated future intent" vs explicit prior-use evidence). Both can be defensible — review confidence and reasoning rather than treating labels as ground truth.

## Configuration

Set your OpenAI API key in `.env`:

```
OPENAI_API_KEY=your-api-key-here
```

When reading `results.csv` in pandas, use `keep_default_na=False` — otherwise the category label `None` is interpreted as a null value and appears as `NaN`.
