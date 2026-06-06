"""
End-to-end tests for the Xylo AI document generation pipeline.
Uses a mock LLM so no real API key is required.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data_loader import (
    parse_email_file,
    load_emails,
    load_crm,
    match_clients,
    ClientRecord,
    CRMRecord,
    EmailRecord,
)
from pipeline.chains import EmailClassification
from pipeline.document_generator import DocumentGenerator, _fmt_crm, _fmt_emails


SAMPLE_DATA = Path(__file__).parent.parent / "sample-data"
EMAILS_DIR = SAMPLE_DATA / "emails"
CRM_CSV = SAMPLE_DATA / "crm_export.csv"

# Canned LLM responses used by the mock
MOCK_CLASSIFICATION = EmailClassification(
    intent="invoice_dispute",
    urgency="high",
    summary="Client disputes invoice total, expecting $2,400 not $2,850.",
    key_asks=["Confirm correct invoice total", "Set up ACH payment"],
    sentiment="neutral",
)

MOCK_DOCUMENT = (
    "June 6, 2026\n\n"
    "Dear Ray,\n\n"
    "Thank you for reaching out about Invoice #4471.\n\n"
    "ACCOUNT STATUS\n--------------\n"
    "Your account is active and in good standing.\n\n"
    "ACTION ITEMS — WHAT WE WILL DO\n-------------------------------\n"
    "- Review Invoice #4471 against the service agreement\n"
    "- Confirm the correct total\n\n"
    "ACTION ITEMS — WHAT WE NEED FROM YOU\n--------------------------------------\n"
    "- Confirm the agreed amount in writing\n\n"
    "NEXT STEPS\n----------\n"
    "We will follow up within 1 business day.\n\n"
    "The Account Management Team | Xylo Professional Services"
)


# ── Helper: build a mock DocumentGenerator ──────────────────────────────────

def make_mock_generator(output_dir: str) -> DocumentGenerator:
    """Return a DocumentGenerator with its LLM chains replaced by mocks."""
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")

    with patch("pipeline.document_generator.make_llm"), \
         patch("pipeline.document_generator.build_classify_chain") as mock_cc, \
         patch("pipeline.document_generator.build_document_chain") as mock_dc:

        mock_cc.return_value.invoke = MagicMock(return_value=MOCK_CLASSIFICATION)
        mock_dc.return_value.invoke = MagicMock(return_value=MOCK_DOCUMENT)

        gen = DocumentGenerator(model="mock", output_dir=output_dir)

    # Replace chains on the live instance
    gen.classify_chain = MagicMock()
    gen.classify_chain.invoke = MagicMock(return_value=MOCK_CLASSIFICATION)
    gen.doc_chain = MagicMock()
    gen.doc_chain.invoke = MagicMock(return_value=MOCK_DOCUMENT)

    return gen


# ── 1. Data loader tests ─────────────────────────────────────────────────────

class TestEmailParsing(unittest.TestCase):

    def test_loads_all_emails(self):
        emails = load_emails(str(EMAILS_DIR))
        self.assertEqual(len(emails), 14, "Expected 14 emails in sample-data/emails/")

    def test_email_fields_populated(self):
        emails = load_emails(str(EMAILS_DIR))
        for e in emails:
            self.assertTrue(e.filename, f"Missing filename: {e}")
            self.assertTrue(e.sender_email, f"Missing sender_email in {e.filename}")
            self.assertTrue(e.body.strip(), f"Empty body in {e.filename}")

    def test_email_01_parsed_correctly(self):
        email = parse_email_file(str(EMAILS_DIR / "email_01.txt"))
        self.assertEqual(email.sender_email, "marcyholt88@gmail.com")
        self.assertIn("paperwork", email.body.lower())

    def test_email_02_invoice_dispute(self):
        email = parse_email_file(str(EMAILS_DIR / "email_02.txt"))
        self.assertEqual(email.sender_email, "r.delgado@delgadohvac.net")
        self.assertIn("invoice", email.body.lower())

    def test_email_subject_extracted(self):
        email = parse_email_file(str(EMAILS_DIR / "email_02.txt"))
        self.assertIn("4471", email.subject)

    def test_email_with_no_subject(self):
        email = parse_email_file(str(EMAILS_DIR / "email_04.txt"))
        self.assertEqual(email.subject, "")

    def test_sender_name_parsed(self):
        email = parse_email_file(str(EMAILS_DIR / "email_02.txt"))
        self.assertTrue(email.sender_name, "Expected sender name from quoted header")


class TestCRMLoading(unittest.TestCase):

    def test_loads_all_crm_records(self):
        records = load_crm(str(CRM_CSV))
        self.assertEqual(len(records), 15, "Expected 15 CRM records")

    def test_crm_fields(self):
        records = load_crm(str(CRM_CSV))
        for r in records:
            self.assertTrue(r.client_id)

    def test_emails_normalized_to_lowercase(self):
        records = load_crm(str(CRM_CSV))
        for r in records:
            self.assertEqual(r.email, r.email.lower())

    def test_known_client_fields(self):
        records = load_crm(str(CRM_CSV))
        ray = next(r for r in records if "delgado" in r.email)
        self.assertEqual(ray.name, "Ray Delgado")
        self.assertEqual(ray.company, "Delgado Heating & Air")
        self.assertEqual(ray.value, "2400")

    def test_messy_data_loaded(self):
        records = load_crm(str(CRM_CSV))
        # Client 1006 has no name — should still load
        anon = next(r for r in records if r.client_id == "1006")
        self.assertEqual(anon.name, "")
        self.assertIn("double charge", anon.notes)


# ── 2. Client matching tests ─────────────────────────────────────────────────

class TestClientMatching(unittest.TestCase):

    def setUp(self):
        self.crm = load_crm(str(CRM_CSV))
        self.emails = load_emails(str(EMAILS_DIR))
        self.clients = match_clients(self.crm, self.emails)

    def test_merge_produces_correct_count(self):
        # 15 CRM + some unmatched emails (Sandra Liu has no CRM record)
        self.assertGreaterEqual(len(self.clients), 15)

    def test_ray_delgado_matched(self):
        ray = next(c for c in self.clients if c.email_address == "r.delgado@delgadohvac.net")
        self.assertIsNotNone(ray.crm)
        self.assertEqual(len(ray.emails), 1)
        self.assertIn("4471", ray.emails[0].subject)

    def test_marcy_holt_matched(self):
        marcy = next(c for c in self.clients if c.email_address == "marcyholt88@gmail.com")
        self.assertIsNotNone(marcy.crm)
        self.assertEqual(len(marcy.emails), 1)

    def test_crm_only_clients_present(self):
        crm_only = [c for c in self.clients if not c.emails]
        names = [c.display_name for c in crm_only]
        self.assertIn("Hank Olson", names)

    def test_email_only_client_present(self):
        # Sandra Liu sent a referral email but has no CRM record
        email_only = next(
            (c for c in self.clients if c.email_address == "sliu@meridianadvisors.co"),
            None
        )
        self.assertIsNotNone(email_only)
        self.assertIsNone(email_only.crm)

    def test_display_name_fallback(self):
        # Anon client (1006) has no name — should fall back to email
        anon = next(c for c in self.clients if c.email_address == "angryclient@protonmail.com")
        self.assertTrue(anon.display_name)

    def test_14_clients_have_email(self):
        with_email = [c for c in self.clients if c.emails]
        self.assertEqual(len(with_email), 14)


# ── 3. Formatter tests ───────────────────────────────────────────────────────

class TestFormatters(unittest.TestCase):

    def test_fmt_crm_none(self):
        result = _fmt_crm(None)
        self.assertIn("No CRM record", result)

    def test_fmt_crm_populated(self):
        crm = CRMRecord(
            client_id="1002", name="Ray Delgado", company="Delgado Heating & Air",
            email="r.delgado@delgadohvac.net", phone="(952) 555-0177",
            status="Active", last_contact="4/29/2024", value="2400",
            notes="invoice dispute open",
        )
        result = _fmt_crm(crm)
        self.assertIn("Ray Delgado", result)
        self.assertIn("2400", result)
        self.assertIn("invoice dispute", result)

    def test_fmt_emails_empty(self):
        result = _fmt_emails([])
        self.assertIn("no inbound", result.lower())

    def test_fmt_emails_multiple(self):
        emails = load_emails(str(EMAILS_DIR))[:3]
        result = _fmt_emails(emails)
        self.assertEqual(result.count("─" * 50), 2)  # 2 dividers for 3 emails


# ── 4. Document generation tests (mocked LLM) ────────────────────────────────

class TestDocumentGenerator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = make_mock_generator(self.tmp)
        self.crm = load_crm(str(CRM_CSV))
        self.emails = load_emails(str(EMAILS_DIR))
        self.clients = match_clients(self.crm, self.emails)
        self.clients_with_email = [c for c in self.clients if c.emails]

    def test_classify_returns_schema(self):
        client = self.clients_with_email[0]
        result = self.gen.classify(client)
        self.assertIsInstance(result, EmailClassification)
        self.assertIn(result.intent, [
            "complaint", "invoice_dispute", "urgent_request", "scheduling",
            "onboarding", "new_lead", "payment_confirmation", "document_request",
            "general_inquiry", "referral", "no_email",
        ])
        self.assertIn(result.urgency, ["critical", "high", "normal", "low"])

    def test_classify_no_email_returns_default(self):
        crm_only = next(c for c in self.clients if not c.emails)
        result = self.gen.classify(crm_only)
        self.assertEqual(result.intent, "no_email")
        self.assertEqual(result.urgency, "low")

    def test_generate_document_returns_string(self):
        client = self.clients_with_email[0]
        classification = self.gen.classify(client)
        doc = self.gen.generate_document(client, classification)
        self.assertIsInstance(doc, str)
        self.assertGreater(len(doc), 50)

    def test_save_writes_file(self):
        client = self.clients_with_email[1]  # Ray Delgado
        classification = MOCK_CLASSIFICATION
        doc = MOCK_DOCUMENT
        filepath = self.gen.save(client, classification, doc)
        self.assertTrue(os.path.exists(filepath))
        content = open(filepath).read()
        self.assertIn("XYLO PROFESSIONAL SERVICES", content)
        self.assertIn("Ray Delgado", content)
        self.assertIn("invoice_dispute", content)

    def test_save_filename_safe(self):
        client = self.clients_with_email[0]
        filepath = self.gen.save(client, MOCK_CLASSIFICATION, MOCK_DOCUMENT)
        filename = os.path.basename(filepath)
        self.assertNotIn(" ", filename)
        self.assertTrue(filename.endswith(".txt"))

    def test_process_one_success(self):
        client = self.clients_with_email[0]
        result = self.gen.process_one(client)
        self.assertEqual(result["status"], "ok")
        self.assertIn("intent", result)
        self.assertIn("urgency", result)
        self.assertTrue(os.path.exists(result["output_file"]))

    def test_process_one_error_handled(self):
        self.gen.classify_chain.invoke.side_effect = RuntimeError("API down")
        client = self.clients_with_email[0]
        result = self.gen.process_one(client)
        self.assertEqual(result["status"], "error")
        self.assertIn("API down", result["error"])

    def test_run_all_generates_one_per_client(self):
        self.gen.classify_chain.invoke.side_effect = None
        self.gen.classify_chain.invoke.return_value = MOCK_CLASSIFICATION
        results = self.gen.run_all(self.clients, with_emails_only=True, verbose=False)
        ok = [r for r in results if r["status"] == "ok"]
        self.assertEqual(len(ok), 14)

    def test_run_all_files_exist(self):
        results = self.gen.run_all(self.clients, with_emails_only=True, verbose=False)
        for r in results:
            if r["status"] == "ok":
                self.assertTrue(os.path.exists(r["output_file"]))

    def test_run_all_include_crm_only(self):
        results = self.gen.run_all(self.clients, with_emails_only=False, verbose=False)
        self.assertEqual(len(results), len(self.clients))

    def test_document_contains_header_metadata(self):
        client = self.clients_with_email[1]  # Ray Delgado
        result = self.gen.process_one(client)
        content = open(result["output_file"]).read()
        self.assertIn("CLIENT", content)
        self.assertIn("GENERATED", content)
        self.assertIn("URGENCY", content)


# ── 5. Full pipeline smoke test ───────────────────────────────────────────────

class TestFullPipelineSampleData(unittest.TestCase):
    """Runs the complete pipeline against all sample data with a mocked LLM."""

    def test_all_14_email_clients_produce_documents(self):
        tmp = tempfile.mkdtemp()
        gen = make_mock_generator(tmp)
        crm = load_crm(str(CRM_CSV))
        emails = load_emails(str(EMAILS_DIR))
        clients = match_clients(crm, emails)

        results = gen.run_all(clients, with_emails_only=True, verbose=False)

        self.assertEqual(len(results), 14)
        for r in results:
            self.assertEqual(r["status"], "ok", f"Failed for {r['client']}: {r.get('error')}")
            self.assertIn("intent", r)
            self.assertIn("urgency", r)
            self.assertTrue(os.path.exists(r["output_file"]))

    def test_known_trade_contractor_clients_processed(self):
        tmp = tempfile.mkdtemp()
        gen = make_mock_generator(tmp)
        crm = load_crm(str(CRM_CSV))
        emails = load_emails(str(EMAILS_DIR))
        clients = match_clients(crm, emails)

        results = gen.run_all(clients, with_emails_only=True, verbose=False)
        processed_emails = {r["email"] for r in results if r["status"] == "ok"}

        # Core trade contractor clients
        self.assertIn("r.delgado@delgadohvac.net", processed_emails)       # HVAC
        self.assertIn("mike@twincitiesflooring.com", processed_emails)     # Flooring
        self.assertIn("dwight.s@stellaroofing.com", processed_emails)      # Roofing

    def test_output_files_have_content(self):
        tmp = tempfile.mkdtemp()
        gen = make_mock_generator(tmp)
        crm = load_crm(str(CRM_CSV))
        emails = load_emails(str(EMAILS_DIR))
        clients = match_clients(crm, emails)

        results = gen.run_all(clients, with_emails_only=True, verbose=False)

        for r in results:
            if r["status"] == "ok":
                size = os.path.getsize(r["output_file"])
                self.assertGreater(size, 100, f"Document too small: {r['output_file']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
