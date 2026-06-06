 ---
  Xylo AI — Run It Yourself

  No hosted version. This is a local Python CLI. Here's everything you need:

  ---
  1. Prerequisites

  - Python 3.10+
  - A Google Gemini API key → get one free at https://aistudio.google.com/apikey

  ---
  2. Install

  cd C:\Programming\Xylo_AI
  pip install -r requirements.txt

  ---
  3. Configure

  # Edit .env and set your key
  GOOGLE_API_KEY=AIza...your-key-here...

  ---
  4. Demo path (short)

  # One client, see the output
  python main.py --limit 1 --model gemini-2.5-flash --summary

  # Single specific client by email
  python main.py --client marcyholt88@gmail.com --model gemini-2.5-flash

  # All 14 clients with emails (add delay to avoid rate limits)
  python main.py --model gemini-2.5-flash --delay 5

  # All 16 including CRM-only clients
  python main.py --all --model gemini-2.5-flash --delay 5

  Output documents land in C:\Programming\Xylo_AI\outputs\ as .txt files named firstname_lastname_intent.txt.

  ---
  What it does

  1. Loads sample-data/crm_export.csv + sample-data/emails/*.txt
  2. Matches emails to CRM records by sender address
  3. Classifies each client (intent + urgency + sentiment) via Gemini
  4. Generates a professional client-ready response document for each

  ---
  Fixed bug: requirements.txt had langchain-anthropic instead of langchain-google-genai — corrected above.