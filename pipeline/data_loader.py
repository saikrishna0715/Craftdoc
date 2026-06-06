import csv
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmailRecord:
    filename: str
    sender_email: str
    sender_name: str
    subject: str
    body: str
    raw: str


@dataclass
class CRMRecord:
    client_id: str
    name: str
    company: str
    email: str
    phone: str
    status: str
    last_contact: str
    value: str
    notes: str


@dataclass
class ClientRecord:
    crm: Optional[CRMRecord]
    emails: list = field(default_factory=list)  # list[EmailRecord]

    @property
    def display_name(self) -> str:
        if self.crm and self.crm.name:
            return self.crm.name
        if self.emails:
            name = self.emails[0].sender_name
            return name if name else self.emails[0].sender_email
        return "Unknown Client"

    @property
    def email_address(self) -> str:
        if self.crm and self.crm.email:
            return self.crm.email
        if self.emails:
            return self.emails[0].sender_email
        return ""


def _parse_sender(sender_line: str):
    """Extract (name, email) from a From: header value."""
    match = re.search(r'<([^>]+)>', sender_line)
    if match:
        email = match.group(1).strip()
        name = re.sub(r'<[^>]+>', '', sender_line).strip().strip('"').strip("'")
    else:
        email = sender_line.strip()
        name = ""
    return name, email.lower()


def parse_email_file(filepath: str) -> EmailRecord:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    sender_line = ""
    subject_line = ""
    body_lines = []
    header_done = False

    for line in raw.split("\n"):
        stripped = line.strip()
        if not header_done:
            if stripped.startswith("From:"):
                sender_line = stripped[5:].strip()
            elif stripped.startswith("Subject:"):
                subject_line = stripped[8:].strip()
            elif stripped == "" and sender_line:
                header_done = True
        else:
            body_lines.append(line)

    sender_name, sender_email = _parse_sender(sender_line)
    body = "\n".join(body_lines).strip()

    return EmailRecord(
        filename=os.path.basename(filepath),
        sender_email=sender_email,
        sender_name=sender_name,
        subject=subject_line,
        body=body,
        raw=raw,
    )


def load_emails(emails_dir: str) -> list:
    records = []
    for fname in sorted(os.listdir(emails_dir)):
        if fname.endswith(".txt"):
            records.append(parse_email_file(os.path.join(emails_dir, fname)))
    return records


def load_crm(csv_path: str) -> list:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(CRMRecord(
                client_id=row.get("client_id", "").strip(),
                name=row.get("name", "").strip(),
                company=row.get("company", "").strip(),
                email=row.get("email", "").strip().lower(),
                phone=row.get("phone", "").strip(),
                status=row.get("status", "").strip(),
                last_contact=row.get("last_contact", "").strip(),
                value=row.get("value", "").strip(),
                notes=row.get("notes", "").strip(),
            ))
    return records


def match_clients(crm_records: list, email_records: list) -> list:
    """
    Merge CRM records and emails into ClientRecord objects.
    Matching is done by email address. Unmatched emails become orphan records.
    """
    email_map: dict = {}
    for er in email_records:
        key = er.sender_email.lower()
        email_map.setdefault(key, []).append(er)

    seen: set = set()
    clients = []

    for crm in crm_records:
        matched_emails = email_map.get(crm.email.lower(), []) if crm.email else []
        clients.append(ClientRecord(crm=crm, emails=matched_emails))
        if crm.email:
            seen.add(crm.email.lower())

    for addr, emails in email_map.items():
        if addr not in seen:
            clients.append(ClientRecord(crm=None, emails=emails))

    return clients
