# CraftDoc— Client Document Generator

Reads raw CRM records and inbound client emails, then uses Google Gemini to classify each client's intent and generate a professional, client-ready response document.

Built for specialty trade contractors: flooring, HVAC, roofing, general contracting.

---

## Requirements

- Python 3.10 or higher
- A Google Gemini API key — get one free at https://aistudio.google.com/apikey

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set your API key**

Create a `.env` file in the project root (copy from the example):

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
GOOGLE_API_KEY=AIzaSy...your-key-here
```

---

## Running

**Quickstart — process one client using the built-in sample data:**

```bash
python main.py --limit 1 --model gemini-2.5-flash
```

**Process all clients that have inbound email:**

```bash
python main.py --model gemini-2.5-flash
```

**Process all clients including CRM-only (no email):**

```bash
python main.py --all --model gemini-2.5-flash
```

**Target a single client by email address:**

```bash
python main.py --client marcyholt88@gmail.com --model gemini-2.5-flash
```

**Use your own data files:**

```bash
python main.py --crm path/to/your/clients.csv --emails path/to/your/inbox/
```

**Add a delay between clients to avoid API rate limits:**

```bash
python main.py --model gemini-2.5-flash --delay 5
```

**Print a JSON summary of all results after processing:**

```bash
python main.py --limit 3 --model gemini-2.5-flash --summary
```

---

## All Options

| Flag | Default | Description |
|---|---|---|
| `--crm FILE` | `sample-data/crm_export.csv` | Path to your CRM CSV file |
| `--emails DIR` | `sample-data/emails/` | Path to your emails directory |
| `--client EMAIL` | — | Process only this one client |
| `--all` | off | Include CRM-only clients with no inbound email |
| `--limit N` | 0 (no limit) | Stop after N clients |
| `--delay SEC` | 0 | Pause between clients (helps with rate limits) |
| `--model MODEL` | `gemini-2.0-flash` | Gemini model ID to use |
| `--output-dir DIR` | `outputs/` | Where to save generated documents |
| `--summary` | off | Print JSON result summary to stdout |

---

## Input Format

**CRM CSV** - must have at minimum an `email` column. All other fields (name, company, status, notes, etc.) are optional but improve output quality.

**Emails directory** - plain `.txt` files, one per email. Each file should start with `From:` and `Subject:` headers followed by the message body:

```
From: client name <client@example.com>
Subject: re: invoice

Message body here...
```

---

## Output

Documents are saved to the `outputs/` directory as `.txt` files named `firstname_lastname_intent.txt`.

Each document includes:
- Account status summary
- Action items (what we will do / what we need from the client)
- Next steps with a realistic timeline

---

## Recommended Model

Use `gemini-2.5-flash` for best results. The default `gemini-2.0-flash` may hit free-tier quota limits quickly.
