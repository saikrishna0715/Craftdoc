import os
import time
from datetime import date

from .data_loader import ClientRecord, CRMRecord, EmailRecord
from .chains import EmailClassification, make_llm, build_classify_chain, build_document_chain  # noqa: F401


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_crm(crm: CRMRecord | None) -> str:
    if not crm:
        return "No CRM record — unknown / new contact."
    parts = [
        f"ID: {crm.client_id}",
        f"Name: {crm.name or '(unnamed)'}",
        f"Company: {crm.company or '(none)'}",
        f"Email: {crm.email}",
        f"Phone: {crm.phone or '(none)'}",
        f"Status: {crm.status or 'unknown'}",
        f"Value: ${crm.value or 'unknown'}",
        f"Last Contact: {crm.last_contact or 'never'}",
        f"Notes: {crm.notes or 'none'}",
    ]
    return " | ".join(parts)


def _fmt_emails(emails: list) -> str:
    if not emails:
        return "(no inbound emails for this client)"
    sections = []
    for e in emails:
        sections.append(
            f"[{e.filename}]\n"
            f"From: {e.sender_name or e.sender_email} <{e.sender_email}>\n"
            f"Subject: {e.subject or '(no subject)'}\n"
            f"\n{e.body}"
        )
    return "\n\n" + ("─" * 50 + "\n\n").join(sections)


def _safe_filename(name: str, intent: str) -> str:
    safe = name.lower().replace(" ", "_")
    safe = "".join(c if c.isalnum() or c == "_" else "" for c in safe)
    return f"{safe}_{intent}.txt"


# ── Document generator ────────────────────────────────────────────────────────

class DocumentGenerator:
    def __init__(self, model: str = "gemini-2.0-flash", output_dir: str = "outputs"):
        self.llm = make_llm(model=model)
        self.classify_chain = build_classify_chain(self.llm)
        self.doc_chain = build_document_chain(self.llm)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def classify(self, record: ClientRecord) -> EmailClassification:
        """Run intent classification. Returns a default for CRM-only records."""
        if not record.emails:
            return EmailClassification(
                intent="no_email",
                urgency="low",
                summary="No inbound email. Proactive follow-up based on CRM status.",
                key_asks=[],
                sentiment="neutral",
            )
        return self.classify_chain.invoke({
            "crm_data": _fmt_crm(record.crm),
            "email_data": _fmt_emails(record.emails),
        })

    def generate_document(self, record: ClientRecord, classification: EmailClassification) -> str:
        crm = record.crm
        return self.doc_chain.invoke({
            "display_name": record.display_name,
            "company": crm.company if crm else "",
            "crm_status": crm.status if crm else "unknown",
            "value": crm.value if crm else "unknown",
            "last_contact": crm.last_contact if crm else "unknown",
            "crm_notes": crm.notes if crm else "",
            "intent": classification.intent,
            "urgency": classification.urgency,
            "summary": classification.summary,
            "key_asks": ", ".join(classification.key_asks) if classification.key_asks else "none",
            "sentiment": classification.sentiment,
            "email_text": _fmt_emails(record.emails),
        })

    def save(self, record: ClientRecord, classification: EmailClassification, document: str) -> str:
        filename = _safe_filename(record.display_name, classification.intent)
        filepath = os.path.join(self.output_dir, filename)

        meta = (
            f"{'=' * 60}\n"
            f"XYLO PROFESSIONAL SERVICES — CLIENT DOCUMENT\n"
            f"{'=' * 60}\n"
            f"CLIENT   : {record.display_name}\n"
            f"EMAIL    : {record.email_address}\n"
            f"COMPANY  : {record.crm.company if record.crm else ''}\n"
            f"INTENT   : {classification.intent}\n"
            f"URGENCY  : {classification.urgency.upper()}\n"
            f"GENERATED: {date.today().isoformat()}\n"
            f"{'=' * 60}\n\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(meta + document)

        return filepath

    def process_one(self, record: ClientRecord) -> dict:
        """Classify + generate + save for a single client. Returns a result dict."""
        try:
            classification = self.classify(record)
            document = self.generate_document(record, classification)
            filepath = self.save(record, classification, document)
            return {
                "client": record.display_name,
                "email": record.email_address,
                "intent": classification.intent,
                "urgency": classification.urgency,
                "summary": classification.summary,
                "output_file": filepath,
                "status": "ok",
            }
        except Exception as exc:
            return {
                "client": record.display_name,
                "email": record.email_address,
                "status": "error",
                "error": str(exc),
            }

    def run_all(
        self,
        records: list,
        with_emails_only: bool = True,
        verbose: bool = True,
        limit: int = 0,
        delay: float = 0.0,
    ) -> list:
        targets = [r for r in records if r.emails] if with_emails_only else records
        if limit and limit > 0:
            targets = targets[:limit]

        print(f"\nGenerating documents for {len(targets)} client(s)...\n")
        results = []

        for i, record in enumerate(targets, 1):
            if verbose:
                print(f"[{i:02d}/{len(targets):02d}] {record.display_name} <{record.email_address}>")

            result = self.process_one(record)
            results.append(result)

            if verbose:
                if result["status"] == "ok":
                    print(f"         [{result['urgency'].upper():8}] {result['intent']}")
                    print(f"         -> {result['output_file']}")
                else:
                    print(f"         [ERROR] {result['error']}")
                print()

            if delay > 0 and i < len(targets):
                if verbose:
                    print(f"         Waiting {delay}s before next client...\n")
                time.sleep(delay)

        ok = sum(1 for r in results if r["status"] == "ok")
        errors = len(results) - ok
        print(f"Complete: {ok} document(s) saved, {errors} error(s).")

        return results
