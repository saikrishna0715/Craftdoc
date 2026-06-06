from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Pydantic schema for structured email classification ──────────────────────

class EmailClassification(BaseModel):
    intent: str = Field(
        description=(
            "One of: complaint, invoice_dispute, urgent_request, scheduling, "
            "onboarding, new_lead, payment_confirmation, document_request, "
            "general_inquiry, referral, no_email"
        )
    )
    urgency: str = Field(description="One of: critical, high, normal, low")
    summary: str = Field(description="One sentence describing what the client needs")
    key_asks: list[str] = Field(description="Specific items the client is requesting or asking about")
    sentiment: str = Field(description="One of: frustrated, neutral, positive, anxious")


# ── Prompts ──────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are an intake specialist at a professional services firm serving specialty \
trade contractors (flooring, HVAC, roofing, contracting).

Analyze the client email and CRM record and classify the interaction.\
"""

_CLASSIFY_HUMAN = """\
CRM RECORD:
{crm_data}

INBOUND EMAIL(S):
{email_data}
"""

_DOCUMENT_SYSTEM = """\
You are a senior account manager at a professional services firm that handles \
bookkeeping, financial reporting, and compliance for specialty trade contractors \
(flooring, HVAC, roofing, general contracting).

Generate a professional, client-ready document to send directly to the client. \
The document should:
- Open with the date and a formal salutation using the client's first name
- Acknowledge their specific situation and what they reached out about
- Include an ACCOUNT STATUS section with their current standing
- Include a clear ACTION ITEMS section split into "What We Will Do" and "What We Need From You"
- Close with NEXT STEPS and a realistic timeline
- Use a warm but professional tone — trade contractors are busy people, so be direct and clear
- Use plain text with ALL CAPS section headers (no markdown, no bullet symbols — use dashes or brackets)

Sign off as: The Account Management Team | Xylo Professional Services\
"""

_DOCUMENT_HUMAN = """\
CLIENT: {display_name}
COMPANY: {company}
CRM STATUS: {crm_status}
ACCOUNT VALUE: {value}
LAST CONTACT: {last_contact}
CRM NOTES: {crm_notes}

EMAIL INTENT: {intent}
URGENCY: {urgency}
SUMMARY: {summary}
KEY ASKS: {key_asks}
CLIENT SENTIMENT: {sentiment}

RAW EMAIL(S):
{email_text}

Write the client-ready document now.\
"""


# ── Chain builders ────────────────────────────────────────────────────────────

def make_llm(model: str = "gemini-2.0-flash", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=2048,
        request_timeout=30,
    )


def build_classify_chain(llm: ChatGoogleGenerativeAI):
    """
    Returns a chain: dict → EmailClassification
    Uses structured output (tool use) for reliable JSON extraction.
    """
    structured_llm = llm.with_structured_output(EmailClassification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _CLASSIFY_SYSTEM),
        ("human", _CLASSIFY_HUMAN),
    ])
    return prompt | structured_llm


def build_document_chain(llm: ChatGoogleGenerativeAI):
    """Returns a chain: dict → str (the generated document text)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _DOCUMENT_SYSTEM),
        ("human", _DOCUMENT_HUMAN),
    ])
    return prompt | llm | StrOutputParser()
