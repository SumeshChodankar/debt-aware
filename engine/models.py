from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class LegalAction(str, Enum):
    """
    Actions the agent can take to protect a borrower.
    Each maps to a real-world legal response under Indian law.
    Updated for RBI Digital Lending Directions 2025.
    """
    SEND_WRITTEN_NOTICE      = "send_written_notice"       # Formal notice to lender's Nodal Officer
    FILE_RBI_COMPLAINT       = "file_rbi_complaint"        # CMS portal complaint (cms.rbi.org.in)
    FILE_POLICE_COMPLAINT    = "file_police_complaint"     # IPC Section 506 / IT Act 66E
    REQUEST_DEBT_VALIDATION  = "request_debt_validation"   # Demand proof of debt & authority to collect
    NEGOTIATE_SETTLEMENT     = "negotiate_settlement"      # Structured settlement offer to lender
    ESCALATE_TO_OMBUDSMAN    = "escalate_to_ombudsman"     # RBI Ombudsman after 30-day lender non-response
    DOCUMENT_VIOLATIONS      = "document_violations"       # Record calls, screenshots, timestamps
    CONTACT_CONSUMER_FORUM   = "contact_consumer_forum"    # Consumer Protection Act 2019 complaint
    # NEW — RBI Digital Lending Directions 2025
    INVOKE_COOLING_OFF       = "invoke_cooling_off"        # Cancel loan within 3-day cooling-off window
    CITE_KFS_VIOLATION       = "cite_kfs_violation"        # Dispute charges not disclosed in Key Fact Statement


class Observation(BaseModel):
    """What the agent sees at each step — the borrower's current situation."""
    turn: int
    harassment_level: float       # 0.0 (none) → 1.0 (severe / threatening)
    debt_amount: float            # Outstanding amount in ₹
    days_overdue: int             # Days since first default
    lender_type: str              # "bank" | "nbfc" | "mfi" | "illegal_app"
    violations_documented: int    # Number of RBI violations logged so far
    complaint_filed: bool         # Whether formal complaint exists
    ombudsman_eligible: bool      # 30-day lender non-response elapsed
    cibil_impact_risk: str        # "low" | "medium" | "high"
    last_agent_action_result: str # What happened after last action
    # NEW — 2025 state fields
    kfs_provided: bool = False    # Whether lender gave Key Fact Statement
    within_cooling_off: bool = False  # Whether 3-day cancellation window is still open
    days_since_disbursement: int = 0  # Days since loan was disbursed


class Action(BaseModel):
    thought_process: str = Field(..., description="Agent's reasoning citing relevant RBI rule or IPC section")
    legal_action: LegalAction
    message_to_lender: str = Field(..., description="Actual communication drafted for the borrower")
    cited_regulation: str  = Field(..., description="Specific RBI circular, IPC section, or Consumer Act clause cited")


class Reward(BaseModel):
    score: float   # Strictly in (0, 1)
    details: str
