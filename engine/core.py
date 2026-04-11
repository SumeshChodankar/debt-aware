import json
import os
import asyncio
from openai import AsyncOpenAI
from .models import Observation, Action, Reward, LegalAction
from .tasks import RBIGrader

# ---------------------------------------------------------------------------
# Legal action → harassment impact mapping
# Some actions reduce harassment directly; others build the legal case.
# ---------------------------------------------------------------------------
ACTION_HARASSMENT_DELTA: dict[str, float] = {
    LegalAction.SEND_WRITTEN_NOTICE:     -0.20,   # Formal notice often stops calls
    LegalAction.FILE_RBI_COMPLAINT:      -0.15,   # Filing creates paper trail, may slow harasser
    LegalAction.FILE_POLICE_COMPLAINT:   -0.30,   # Police complaint has strongest deterrent effect
    LegalAction.REQUEST_DEBT_VALIDATION: -0.10,   # Validation request may pause collection
    LegalAction.NEGOTIATE_SETTLEMENT:    -0.05,   # Negotiation shows willingness — minor relief
    LegalAction.ESCALATE_TO_OMBUDSMAN:  -0.25,   # Ombudsman escalation triggers bank response
    LegalAction.DOCUMENT_VIOLATIONS:     +0.00,   # Documentation alone doesn't stop harassment
    LegalAction.CONTACT_CONSUMER_FORUM:  -0.20,   # Consumer forum filing creates regulatory pressure
    LegalAction.INVOKE_COOLING_OFF:      -0.35,   # 2025: cancellation within 3 days — strong deterrent
    LegalAction.CITE_KFS_VIOLATION:      -0.15,   # 2025: KFS dispute creates regulatory pressure
}

# Actions that build the complaint trail
COMPLAINT_ACTIONS = {
    LegalAction.FILE_RBI_COMPLAINT,
    LegalAction.ESCALATE_TO_OMBUDSMAN,
    LegalAction.FILE_POLICE_COMPLAINT,
    LegalAction.CONTACT_CONSUMER_FORUM,
    LegalAction.CITE_KFS_VIOLATION,   # 2025: KFS dispute is a formal complaint action
}

# Actions that document violations
DOCUMENTATION_ACTIONS = {
    LegalAction.DOCUMENT_VIOLATIONS,
    LegalAction.SEND_WRITTEN_NOTICE,
    LegalAction.REQUEST_DEBT_VALIDATION,
}

# Reward table — all strictly in (0, 1)
ACTION_REWARD: dict[str, float] = {
    LegalAction.SEND_WRITTEN_NOTICE:     0.70,
    LegalAction.FILE_RBI_COMPLAINT:      0.80,
    LegalAction.FILE_POLICE_COMPLAINT:   0.75,
    LegalAction.REQUEST_DEBT_VALIDATION: 0.65,
    LegalAction.NEGOTIATE_SETTLEMENT:    0.70,
    LegalAction.ESCALATE_TO_OMBUDSMAN:  0.85,
    LegalAction.DOCUMENT_VIOLATIONS:    0.55,
    LegalAction.CONTACT_CONSUMER_FORUM:  0.72,
    LegalAction.INVOKE_COOLING_OFF:      0.88,  # 2025: highest value when within window
    LegalAction.CITE_KFS_VIOLATION:      0.78,  # 2025: strong if KFS was not provided
}

# Penalty for illegal app lenders — some actions are invalid
ILLEGAL_APP_INVALID_ACTIONS = {
    LegalAction.NEGOTIATE_SETTLEMENT,   # No legal entity to negotiate with
}

# ---------------------------------------------------------------------------
# Load knowledge base (Indian legal context for RAG)
# ---------------------------------------------------------------------------
def _load_kb() -> dict:
    kb_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base.json")
    try:
        with open(os.path.abspath(kb_path), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

LEGAL_KB: dict = _load_kb()


def _build_legal_context(action: LegalAction, lender_type: str) -> str:
    """Return the most relevant Indian law snippets for this action + lender type."""
    snippets = []
    kb = LEGAL_KB

    if action in (LegalAction.FILE_RBI_COMPLAINT, LegalAction.ESCALATE_TO_OMBUDSMAN):
        if "RBI_Fair_Practices_Code" in kb:
            snippets.append(kb["RBI_Fair_Practices_Code"])
        if "RBI_Ombudsman" in kb:
            snippets.append(kb["RBI_Ombudsman"])

    if action == LegalAction.FILE_POLICE_COMPLAINT:
        if "IPC_Section_506" in kb:
            snippets.append(kb["IPC_Section_506"])
        if lender_type == "illegal_app" and "IT_Act_66E" in kb:
            snippets.append(kb["IT_Act_66E"])

    if action == LegalAction.REQUEST_DEBT_VALIDATION:
        if "RBI_Recovery_Guidelines" in kb:
            snippets.append(kb["RBI_Recovery_Guidelines"])

    if action == LegalAction.CONTACT_CONSUMER_FORUM:
        if "Consumer_Protection_Act_2019" in kb:
            snippets.append(kb["Consumer_Protection_Act_2019"])

    if not snippets:
        snippets = [f"{k}: {v}" for k, v in kb.items()]

    return "\n".join(snippets)


class RBIRightsEnv:
    """
    RBI Rights Navigator — a mini-game RL environment where an AI agent
    learns to protect Indian borrowers from illegal debt collection by
    choosing the right legal action at each step.

    Four tasks with increasing difficulty:
      easy   — stop harassment (document + written notice)
      medium — file RBI complaint correctly (CMS + Ombudsman path)
      hard   — negotiate settlement while protecting CIBIL
      expert — full illegal app takedown (all four dimensions)
    """

    TASK_DEFAULTS = {
        "easy": {
            "harassment_level": 0.6,
            "debt_amount": 15000.0,
            "days_overdue": 45,
            "lender_type": "nbfc",
            "cibil_impact_risk": "medium",
            "kfs_provided": False,
            "within_cooling_off": False,
            "days_since_disbursement": 45,
        },
        "medium": {
            "harassment_level": 0.7,
            "debt_amount": 45000.0,
            "days_overdue": 75,
            "lender_type": "bank",
            "cibil_impact_risk": "medium",
            "kfs_provided": False,
            "within_cooling_off": False,
            "days_since_disbursement": 75,
        },
        "hard": {
            "harassment_level": 0.5,
            "debt_amount": 120000.0,
            "days_overdue": 120,
            "lender_type": "nbfc",
            "cibil_impact_risk": "high",
            "kfs_provided": False,
            "within_cooling_off": False,
            "days_since_disbursement": 120,
        },
        "expert": {
            "harassment_level": 0.95,
            "debt_amount": 12000.0,
            "days_overdue": 20,
            "lender_type": "illegal_app",
            "cibil_impact_risk": "high",
            "kfs_provided": False,
            "within_cooling_off": False,
            "days_since_disbursement": 20,
        },
        # NEW — 2025 tasks
        "cooling_off": {
            "harassment_level": 0.3,
            "debt_amount": 8000.0,
            "days_overdue": 0,
            "lender_type": "nbfc",
            "cibil_impact_risk": "low",
            "kfs_provided": False,
            "within_cooling_off": True,   # still within 3-day window
            "days_since_disbursement": 1,
        },
        "kfs_violation": {
            "harassment_level": 0.4,
            "debt_amount": 25000.0,
            "days_overdue": 30,
            "lender_type": "nbfc",
            "cibil_impact_risk": "medium",
            "kfs_provided": False,        # lender never gave KFS — the violation
            "within_cooling_off": False,
            "days_since_disbursement": 30,
        },
    }

    MAX_STEPS = {"easy": 5, "medium": 5, "hard": 6, "expert": 7}

    def __init__(self, task_level: str = "easy", initial_data: dict | None = None):
        self.task_level   = task_level
        self.initial_data = initial_data
        self.api_key      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.base_url     = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
        self.model_name   = os.getenv("MODEL_NAME", "gpt-4o")

        # State
        self._state:              dict             = {}
        self.turn:                int              = 0
        self.initial_obs:         Observation|None = None
        self._last_lender_reply:  str              = "No response yet."

    # ------------------------------------------------------------------
    def _get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    async def reset(self, initial_data: dict | None = None) -> Observation:
        data = initial_data or self.initial_data or self.TASK_DEFAULTS.get(self.task_level, self.TASK_DEFAULTS["easy"])
        self._state = dict(data)
        self.turn   = 0
        self._last_lender_reply = "Awaiting your first action."
        self.initial_obs = self._to_obs()
        return self.initial_obs

    def state_snapshot(self) -> Observation:
        return self._to_obs()

    # ------------------------------------------------------------------
    async def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.turn += 1
        la = LegalAction(action.legal_action)

        # Invalid action check (e.g. negotiating with illegal app)
        invalid = (
            self._state["lender_type"] == "illegal_app" and
            la in ILLEGAL_APP_INVALID_ACTIONS
        )

        if invalid:
            reward_val = 0.10   # Near-zero — chose wrong strategy for lender type
            self._last_lender_reply = (
                "This lender is unregistered. Negotiation is not possible — "
                "legal/regulatory action is the only effective path."
            )
        else:
            # Apply legal action effects to state
            self._apply_action(la)
            # Get lender/system response via LLM
            self._last_lender_reply = await self._get_lender_response(action, la)
            reward_val = ACTION_REWARD.get(la, 0.55)

        max_steps = self.MAX_STEPS.get(self.task_level, 5)
        done = (self.turn >= max_steps) or (self._state["harassment_level"] <= 0.05)

        info: dict = {}
        if done and self.initial_obs is not None:
            grader_map = {
                "easy":   RBIGrader.grade_easy,
                "medium": RBIGrader.grade_medium,
                "hard":   RBIGrader.grade_hard,
                "expert": RBIGrader.grade_expert,
            }
            grader_fn = grader_map.get(self.task_level, RBIGrader.grade_medium)
            result = grader_fn(self._to_obs(), self.initial_obs)
            if isinstance(result, dict):
                info["grader_score"]  = float(result["score"])
                info["grader_passed"] = bool(result.get("passed", False))
            else:
                info["grader_score"]  = max(0.001, min(0.999, float(result)))
                info["grader_passed"] = float(result) >= 0.5

        return (
            self._to_obs(),
            Reward(score=reward_val, details=self._last_lender_reply),
            done,
            info,
        )

    async def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------
    def _apply_action(self, la: LegalAction) -> None:
        """Apply the legal action's real-world effects to the borrower's state."""
        s = self._state

        # Harassment reduction
        delta = ACTION_HARASSMENT_DELTA.get(la, 0.0)
        s["harassment_level"] = max(0.0, min(1.0, s["harassment_level"] + delta))

        # Documentation
        if la in DOCUMENTATION_ACTIONS:
            s["violations_documented"] = s.get("violations_documented", 0) + 1

        # Complaint tracking
        if la in COMPLAINT_ACTIONS:
            s["complaint_filed"] = True

        # Ombudsman eligibility — triggered after filing complaint
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("complaint_filed"):
            s["ombudsman_eligible"] = True

        # Debt reduction on settlement
        if la == LegalAction.NEGOTIATE_SETTLEMENT:
            reduction = {"bank": 0.15, "nbfc": 0.25, "mfi": 0.30, "illegal_app": 0.0}
            s["debt_amount"] = s["debt_amount"] * (1 - reduction.get(s["lender_type"], 0.15))

        # Police complaint on illegal app can trigger debt waiver
        if la == LegalAction.FILE_POLICE_COMPLAINT and s["lender_type"] == "illegal_app":
            s["debt_amount"] = s["debt_amount"] * 0.30   # 70% waiver — illegal apps back off
            s["cibil_impact_risk"] = "low"

        # CIBIL impact improvement from successful RBI complaint
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("ombudsman_eligible"):
            if s["cibil_impact_risk"] == "high":
                s["cibil_impact_risk"] = "medium"

        # NEW 2025 — Invoke cooling-off right
        if la == LegalAction.INVOKE_COOLING_OFF:
            if s.get("within_cooling_off"):
                # Lender must cancel — full debt waiver of charges, only principal owed
                s["debt_amount"] = s["debt_amount"] * 0.05  # only proportionate interest remains
                s["harassment_level"] = max(0.0, s["harassment_level"] - 0.35)
                s["complaint_filed"] = True
            else:
                # Window expired — still creates pressure but less effective
                s["debt_amount"] = s["debt_amount"] * 0.90
                s["complaint_filed"] = True

        # NEW 2025 — Cite KFS violation
        if la == LegalAction.CITE_KFS_VIOLATION:
            if not s.get("kfs_provided"):
                # Strong position — undisclosed charges unenforceable
                s["debt_amount"] = s["debt_amount"] * 0.70  # 30% reduction on disputed charges
                s["complaint_filed"] = True
                s["violations_documented"] = s.get("violations_documented", 0) + 1
            else:
                # KFS was provided — weaker argument
                s["violations_documented"] = s.get("violations_documented", 0) + 1

    # ------------------------------------------------------------------
    # LLM — adversarial lender/system response
    # ------------------------------------------------------------------
    async def _get_lender_response(self, action: Action, la: LegalAction) -> str:
        legal_ctx = _build_legal_context(la, self._state["lender_type"])
        lender_type = self._state["lender_type"]
        harassment  = self._state["harassment_level"]

        persona = {
            "bank":         f"a bank's collections department (moderately cooperative, harassment level {harassment:.0%})",
            "nbfc":         f"an NBFC recovery agent (aggressive, harassment level {harassment:.0%})",
            "mfi":          f"an MFI field officer (persistent, visiting weekly, harassment level {harassment:.0%})",
            "illegal_app":  f"an unregistered app's agent (threatening, using personal data, harassment level {harassment:.0%})",
        }.get(lender_type, "a lender")

        prompt = (
            f"You are {persona} responding to a borrower's legal action.\n"
            f"Relevant Indian law context:\n{legal_ctx}\n\n"
            f"Borrower's action: {la.value}\n"
            f"Borrower's message: \"{action.message_to_lender}\"\n"
            f"Regulation cited: {action.cited_regulation}\n\n"
            f"Respond realistically as this lender type would. "
            f"If the action is legally sound, the lender becomes more cooperative. "
            f"If the action cites the wrong law or is weak, the lender may dismiss it. "
            f"Keep response under 80 words. Be realistic to Indian debt recovery culture."
        )

        client = self._get_client()

        # Canned realistic responses per lender type — used as fallback
        canned = {
            "bank":        "Your complaint has been noted. We will review within 30 days as per RBI guidelines.",
            "nbfc":        "We acknowledge your notice. Our team will contact you to discuss repayment options.",
            "mfi":         "Your request has been received. Our field officer will follow up accordingly.",
            "illegal_app": "Your payment is overdue. Please pay immediately to avoid further action.",
        }
        fallback = canned.get(self._state.get("lender_type", "bank"), canned["bank"])

        for attempt in range(3):
            try:
                res = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=100,  # keep short — lender response only
                )
                return res.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    import asyncio as _asyncio
                    wait = (2 ** attempt) * 8  # 8s, 16s, 32s
                    print(f"    [lender LLM rate limit] sleeping {wait}s (attempt {attempt+1}/3)...",
                          flush=True)
                    await _asyncio.sleep(wait)
                else:
                    # Non-rate-limit error — use canned immediately
                    return fallback
        # All retries exhausted — use canned response to keep episode alive
        return fallback

    # ------------------------------------------------------------------
    def _to_obs(self) -> Observation:
        s = self._state
        return Observation(
            turn                     = self.turn,
            harassment_level         = round(s.get("harassment_level", 0.5), 3),
            debt_amount              = round(s.get("debt_amount", 0.0), 2),
            days_overdue             = int(s.get("days_overdue", 0)),
            lender_type              = s.get("lender_type", "bank"),
            violations_documented    = int(s.get("violations_documented", 0)),
            complaint_filed          = bool(s.get("complaint_filed", False)),
            ombudsman_eligible       = bool(s.get("ombudsman_eligible", False)),
            cibil_impact_risk        = s.get("cibil_impact_risk", "medium"),
            last_agent_action_result = self._last_lender_reply,
            kfs_provided             = bool(s.get("kfs_provided", False)),
            within_cooling_off       = bool(s.get("within_cooling_off", False)),
            days_since_disbursement  = int(s.get("days_since_disbursement", 0)),
        )
