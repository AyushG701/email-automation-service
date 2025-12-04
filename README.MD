# Email Automation Service

# Initial step

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-cache-dir -r requirements.txt
deactivate
```

# Server start

## Development mode

```
fastapi dev main.py
uvicorn main:app --reload --host 0.0.0.0 --port 7153
```

# Production mode

```
fastapi run main.py
```

## FEATURES

1. Read email from email ingestion queue.
2. Classify the email.
3. Find intent from email for further actions.

# Actions type:

1. Load Offer action.
2. RC Processing action.
3. Negotiation action.
