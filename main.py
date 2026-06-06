"""
Xylo AI — Client Document Generator
Specialty trade contractors: flooring, HVAC, roofing

Usage:
  python main.py                                        # use built-in sample data
  python main.py --crm path/to/crm.csv                 # custom CRM file
  python main.py --emails path/to/emails/              # custom emails directory
  python main.py --crm my.csv --emails my_emails/      # fully custom input
  python main.py --all                                  # include CRM-only clients (no email)
  python main.py --client foo@bar.com                  # single client by email
  python main.py --limit 5                             # process first 5 clients only
  python main.py --limit 5 --delay 10                  # process 5 clients, 10s pause between each
  python main.py --summary                             # also print JSON summary to stdout
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pipeline.data_loader import load_crm, load_emails, match_clients
from pipeline.document_generator import DocumentGenerator


BASE_DIR = Path(__file__).parent
SAMPLE_DATA = BASE_DIR / "sample-data"
DEFAULT_OUTPUT = BASE_DIR / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate client-ready documents from raw CRM + email data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--crm", default=str(SAMPLE_DATA / "crm_export.csv"), metavar="FILE", help="Path to CRM CSV file (default: sample-data/crm_export.csv)")
    parser.add_argument("--emails", default=str(SAMPLE_DATA / "emails"), metavar="DIR", help="Path to emails directory (default: sample-data/emails/)")
    parser.add_argument("--client", metavar="EMAIL", help="Process only this client email address")
    parser.add_argument("--all", action="store_true", help="Include CRM-only clients with no inbound email")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="Stop after N clients (0 = no limit)")
    parser.add_argument("--delay", type=float, default=0.0, metavar="SEC", help="Seconds to wait between clients (helps with rate limits)")
    parser.add_argument("--model", default="gemini-2.0-flash", metavar="MODEL", help="Google Gemini model ID (default: gemini-2.0-flash)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), metavar="DIR", help="Output directory")
    parser.add_argument("--summary", action="store_true", help="Print JSON result summary to stdout")
    args = parser.parse_args()

    crm_path = Path(args.crm)
    emails_path = Path(args.emails)

    if not crm_path.is_file():
        print(f"ERROR: CRM file not found: {crm_path}", file=sys.stderr)
        sys.exit(1)
    if not emails_path.is_dir():
        print(f"ERROR: Emails directory not found: {emails_path}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set.", file=sys.stderr)
        print("  Add it to a .env file or export it as an environment variable.", file=sys.stderr)
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading CRM data from:  {crm_path}")
    print(f"Loading emails from:    {emails_path}")

    crm_records = load_crm(crm_path)
    email_records = load_emails(emails_path)
    all_clients = match_clients(crm_records, email_records)

    matched = sum(1 for c in all_clients if c.emails)
    print(f"\n{len(crm_records)} CRM records | {len(email_records)} emails | {len(all_clients)} merged clients")
    print(f"{matched} clients have inbound email, {len(all_clients) - matched} are CRM-only\n")

    # ── Filter ───────────────────────────────────────────────────────────────
    if args.client:
        target = args.client.strip().lower()
        clients = [c for c in all_clients if c.email_address.lower() == target]
        if not clients:
            print(f"ERROR: No client found with email '{target}'", file=sys.stderr)
            sys.exit(1)
    else:
        clients = all_clients

    # ── Generate ─────────────────────────────────────────────────────────────
    generator = DocumentGenerator(model=args.model, output_dir=args.output_dir)
    results = generator.run_all(
        clients,
        with_emails_only=not args.all,
        limit=args.limit,
        delay=args.delay,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\nDocuments saved to: {args.output_dir}")

    if args.summary:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
