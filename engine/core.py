# """
# engine/core.py — RBI Rights Navigator RL Environment

# What makes this genuinely RL (not just planning):

# 1. STOCHASTIC LENDER RESISTANCE
#    The lender LLM returns a structured JSON response that includes
#    a resistance_level and a partial_concession flag. These directly
#    modify the state delta — the same action produces different outcomes
#    depending on what the lender decides. The agent must read and
#    respond to feedback.

# 2. DIMINISHING RETURNS ON REPEATED ACTIONS
#    Using the same legal action twice in a row halves its effect.
#    Using it a third time quarters it. The agent is forced to diversify
#    strategy — which is what real-world legal tactics require.

# 3. LENDER-TYPE-SPECIFIC REWARD TABLE
#    escalate_to_ombudsman scores 0.85 against a bank but 0.20 against
#    an illegal app (which has no RBI standing). The agent must learn
#    which tools work for which adversary.

# 4. LENDER ESCALATION — ACTIONS CAN BACKFIRE
#    Aggressive actions (firm_demand, police complaint) against cooperative
#    lenders can increase harassment. The lender's resistance_level feeds
#    back into the state. The agent must de-escalate when the lender
#    is already cooperating.

# 5. PARTIAL CONCESSIONS
#    The lender can offer partial outcomes — e.g. reduce harassment by 50%
#    of the requested amount. The agent must decide whether to accept or
#    push further. This creates genuine sequential decision-making.
# """

# import json
# import os
# import random
# import asyncio
# from openai import AsyncOpenAI
# from .models import Observation, Action, Reward, LegalAction
# from .tasks import RBIGrader

# # ---------------------------------------------------------------------------
# # Lender-type resistance profiles
# # Each lender starts with a base resistance level and aggression multiplier.
# # These determine how much of the action's effect actually lands.
# # ---------------------------------------------------------------------------
# LENDER_PROFILES = {
#     "bank": {
#         "base_resistance":   0.30,   # moderately cooperative by default
#         "aggression":        0.40,   # escalates moderately when pushed
#         "legal_sensitivity": 0.80,   # responds strongly to valid legal citations
#         "ombudsman_fear":    0.90,   # banks fear Ombudsman awards (₹20 lakh)
#         "max_resistance":    0.70,   # won't go above this no matter what
#     },
#     "nbfc": {
#         "base_resistance":   0.55,   # more resistant than banks
#         "aggression":        0.60,   # escalates more readily
#         "legal_sensitivity": 0.55,   # somewhat aware of legal consequences
#         "ombudsman_fear":    0.65,   # fears Ombudsman but less than banks
#         "max_resistance":    0.85,
#     },
#     "mfi": {
#         "base_resistance":   0.65,   # persistent — field officers under pressure
#         "aggression":        0.70,   # group-pressure model, escalates fast
#         "legal_sensitivity": 0.40,   # rural MFIs less aware of legal rights
#         "ombudsman_fear":    0.50,   # weaker Ombudsman reach in rural areas
#         "max_resistance":    0.90,
#     },
#     "illegal_app": {
#         "base_resistance":   0.90,   # almost no cooperation
#         "aggression":        0.95,   # threatens contacts, escalates maximally
#         "legal_sensitivity": 0.20,   # barely responds to legal citations
#         "ombudsman_fear":    0.10,   # not registered — Ombudsman has no reach
#         "max_resistance":    1.00,   # can hit full resistance
#     },
# }

# # ---------------------------------------------------------------------------
# # Action effect table — BASE values before lender modifiers apply
# # ---------------------------------------------------------------------------
# ACTION_BASE_DELTA: dict[str, float] = {
#     LegalAction.SEND_WRITTEN_NOTICE:     -0.25,
#     LegalAction.FILE_RBI_COMPLAINT:      -0.20,
#     LegalAction.FILE_POLICE_COMPLAINT:   -0.35,
#     LegalAction.REQUEST_DEBT_VALIDATION: -0.12,
#     LegalAction.NEGOTIATE_SETTLEMENT:    -0.08,
#     LegalAction.ESCALATE_TO_OMBUDSMAN:  -0.30,
#     LegalAction.DOCUMENT_VIOLATIONS:    +0.00,
#     LegalAction.CONTACT_CONSUMER_FORUM: -0.22,
#     LegalAction.INVOKE_COOLING_OFF:     -0.40,
#     LegalAction.CITE_KFS_VIOLATION:     -0.18,
# }

# # ---------------------------------------------------------------------------
# # Lender-type specific reward multipliers
# # The base reward is modified by how appropriate the action is for this
# # specific lender type. Using Ombudsman against illegal app = low value.
# # ---------------------------------------------------------------------------
# LENDER_ACTION_MULTIPLIER: dict[str, dict] = {
#     "bank": {
#         LegalAction.ESCALATE_TO_OMBUDSMAN:  1.00,   # highly effective vs bank
#         LegalAction.FILE_RBI_COMPLAINT:     1.00,
#         LegalAction.SEND_WRITTEN_NOTICE:    0.90,
#         LegalAction.NEGOTIATE_SETTLEMENT:   0.85,
#         LegalAction.CONTACT_CONSUMER_FORUM: 0.80,
#         LegalAction.FILE_POLICE_COMPLAINT:  0.60,   # less effective vs bank
#         LegalAction.REQUEST_DEBT_VALIDATION: 0.75,
#         LegalAction.DOCUMENT_VIOLATIONS:    0.70,
#         LegalAction.INVOKE_COOLING_OFF:     0.95,
#         LegalAction.CITE_KFS_VIOLATION:     0.95,
#     },
#     "nbfc": {
#         LegalAction.FILE_RBI_COMPLAINT:     1.00,
#         LegalAction.ESCALATE_TO_OMBUDSMAN:  0.85,
#         LegalAction.NEGOTIATE_SETTLEMENT:   1.00,   # NBFCs negotiate
#         LegalAction.SEND_WRITTEN_NOTICE:    0.85,
#         LegalAction.FILE_POLICE_COMPLAINT:  0.80,
#         LegalAction.CONTACT_CONSUMER_FORUM: 0.75,
#         LegalAction.REQUEST_DEBT_VALIDATION: 0.80,
#         LegalAction.DOCUMENT_VIOLATIONS:    0.70,
#         LegalAction.INVOKE_COOLING_OFF:     0.90,
#         LegalAction.CITE_KFS_VIOLATION:     0.90,
#     },
#     "mfi": {
#         LegalAction.FILE_POLICE_COMPLAINT:  1.00,   # police most effective vs MFI
#         LegalAction.CONTACT_CONSUMER_FORUM: 0.95,
#         LegalAction.FILE_RBI_COMPLAINT:     0.85,
#         LegalAction.DOCUMENT_VIOLATIONS:    0.90,   # documentation key for MFI cases
#         LegalAction.SEND_WRITTEN_NOTICE:    0.70,
#         LegalAction.NEGOTIATE_SETTLEMENT:   0.80,
#         LegalAction.ESCALATE_TO_OMBUDSMAN:  0.65,   # weaker in rural
#         LegalAction.REQUEST_DEBT_VALIDATION: 0.70,
#         LegalAction.INVOKE_COOLING_OFF:     0.60,
#         LegalAction.CITE_KFS_VIOLATION:     0.70,
#     },
#     "illegal_app": {
#         LegalAction.FILE_POLICE_COMPLAINT:  1.00,   # only real option
#         LegalAction.FILE_RBI_COMPLAINT:     0.90,
#         LegalAction.DOCUMENT_VIOLATIONS:    0.85,
#         LegalAction.ESCALATE_TO_OMBUDSMAN:  0.20,   # useless — not registered
#         LegalAction.NEGOTIATE_SETTLEMENT:   0.05,   # invalid — penalised
#         LegalAction.SEND_WRITTEN_NOTICE:    0.40,
#         LegalAction.CONTACT_CONSUMER_FORUM: 0.70,
#         LegalAction.REQUEST_DEBT_VALIDATION: 0.60,
#         LegalAction.INVOKE_COOLING_OFF:     0.50,
#         LegalAction.CITE_KFS_VIOLATION:     0.55,
#     },
# }

# # Base reward table (before lender multiplier)
# BASE_REWARD: dict[str, float] = {
#     LegalAction.ESCALATE_TO_OMBUDSMAN:  0.85,
#     LegalAction.INVOKE_COOLING_OFF:     0.88,
#     LegalAction.FILE_RBI_COMPLAINT:     0.80,
#     LegalAction.FILE_POLICE_COMPLAINT:  0.78,
#     LegalAction.CITE_KFS_VIOLATION:     0.78,
#     LegalAction.CONTACT_CONSUMER_FORUM: 0.72,
#     LegalAction.SEND_WRITTEN_NOTICE:    0.70,
#     LegalAction.NEGOTIATE_SETTLEMENT:   0.70,
#     LegalAction.REQUEST_DEBT_VALIDATION: 0.65,
#     LegalAction.DOCUMENT_VIOLATIONS:    0.55,
# }

# # Keep ACTION_REWARD for backward compatibility with server/main.py
# ACTION_REWARD = BASE_REWARD

# # ---------------------------------------------------------------------------
# # Load knowledge base
# # ---------------------------------------------------------------------------
# def _load_kb() -> dict:
#     kb_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base.json")
#     try:
#         with open(os.path.abspath(kb_path), "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {}

# LEGAL_KB: dict = _load_kb()


# def _build_legal_context(action: LegalAction, lender_type: str) -> str:
#     snippets = []
#     kb = LEGAL_KB
#     if action in (LegalAction.FILE_RBI_COMPLAINT, LegalAction.ESCALATE_TO_OMBUDSMAN):
#         for k in ("RBI_Fair_Practices_Code", "RBI_Ombudsman"):
#             if k in kb: snippets.append(kb[k])
#     if action == LegalAction.FILE_POLICE_COMPLAINT:
#         if "IPC_Section_506" in kb: snippets.append(kb["IPC_Section_506"])
#         if lender_type == "illegal_app" and "IT_Act_66E" in kb:
#             snippets.append(kb["IT_Act_66E"])
#     if action == LegalAction.REQUEST_DEBT_VALIDATION:
#         if "RBI_Recovery_Guidelines" in kb: snippets.append(kb["RBI_Recovery_Guidelines"])
#     if action == LegalAction.CONTACT_CONSUMER_FORUM:
#         if "Consumer_Protection_Act_2019" in kb: snippets.append(kb["Consumer_Protection_Act_2019"])
#     if action in (LegalAction.INVOKE_COOLING_OFF,):
#         if "Cooling_Off_Right" in kb: snippets.append(kb["Cooling_Off_Right"])
#     if action == LegalAction.CITE_KFS_VIOLATION:
#         if "KFS_Violation_Rights" in kb: snippets.append(kb["KFS_Violation_Rights"])
#     if not snippets:
#         snippets = [f"{k}: {v}" for k, v in list(kb.items())[:3]]
#     return "\n".join(snippets)


# # ---------------------------------------------------------------------------
# # COMPLAINT and DOCUMENTATION action sets (used by _apply_action)
# # ---------------------------------------------------------------------------
# COMPLAINT_ACTIONS = {
#     LegalAction.FILE_RBI_COMPLAINT,
#     LegalAction.ESCALATE_TO_OMBUDSMAN,
#     LegalAction.FILE_POLICE_COMPLAINT,
#     LegalAction.CONTACT_CONSUMER_FORUM,
#     LegalAction.CITE_KFS_VIOLATION,
# }

# DOCUMENTATION_ACTIONS = {
#     LegalAction.DOCUMENT_VIOLATIONS,
#     LegalAction.SEND_WRITTEN_NOTICE,
#     LegalAction.REQUEST_DEBT_VALIDATION,
# }


# class RBIRightsEnv:
#     """
#     RBI Rights Navigator — genuine RL environment.

#     Key RL properties:
#     - Stochastic: lender LLM returns resistance_level and partial_concession
#       that directly modify state transitions
#     - Non-deterministic: same action can produce different outcomes
#     - Adaptive adversary: lender resistance changes based on episode history
#     - Diminishing returns: repeated actions lose effectiveness
#     - Lender-specific dynamics: strategy must adapt to lender type
#     """

#     TASK_DEFAULTS = {
#         "easy": {
#             "harassment_level": 0.6, "debt_amount": 15000.0, "days_overdue": 45,
#             "lender_type": "nbfc", "cibil_impact_risk": "medium",
#             "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 45,
#         },
#         "medium": {
#             "harassment_level": 0.7, "debt_amount": 45000.0, "days_overdue": 75,
#             "lender_type": "bank", "cibil_impact_risk": "medium",
#             "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 75,
#         },
#         "hard": {
#             "harassment_level": 0.5, "debt_amount": 120000.0, "days_overdue": 120,
#             "lender_type": "nbfc", "cibil_impact_risk": "high",
#             "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 120,
#         },
#         "expert": {
#             "harassment_level": 0.95, "debt_amount": 12000.0, "days_overdue": 20,
#             "lender_type": "illegal_app", "cibil_impact_risk": "high",
#             "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 20,
#         },
#         "cooling_off": {
#             "harassment_level": 0.3, "debt_amount": 8000.0, "days_overdue": 0,
#             "lender_type": "nbfc", "cibil_impact_risk": "low",
#             "kfs_provided": False, "within_cooling_off": True, "days_since_disbursement": 1,
#         },
#         "kfs_violation": {
#             "harassment_level": 0.4, "debt_amount": 25000.0, "days_overdue": 30,
#             "lender_type": "nbfc", "cibil_impact_risk": "medium",
#             "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 30,
#         },
#     }

#     MAX_STEPS = {"easy": 5, "medium": 5, "hard": 6, "expert": 7,
#                  "cooling_off": 4, "kfs_violation": 5}

#     def __init__(
#         self,
#         task_level: str = "easy",
#         initial_data: dict | None = None,
#         deterministic: bool = False,
#     ):
#         """
#         deterministic=False  → full RL mode: stochastic lender, diminishing returns,
#                                lender-specific multipliers. Use for training.
#         deterministic=True   → submission mode: fixed canned lender responses,
#                                standard reward table. Guarantees consistent scores
#                                for the OpenEnv validator.
#         """
#         self.task_level    = task_level
#         self.initial_data  = initial_data
#         self.deterministic = deterministic
#         self.api_key       = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
#         self.base_url      = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
#         self.model_name    = os.getenv("MODEL_NAME", "gpt-4o")

#         self._state: dict             = {}
#         self.turn:   int              = 0
#         self.initial_obs: Observation | None = None
#         self._last_lender_reply: str  = "No response yet."

#         # RL state — tracks history for diminishing returns and lender adaptation
#         self._action_history: list[str]  = []
#         self._lender_resistance: float   = 0.0
#         self._consecutive_same: int      = 0
#         self._last_action: str | None    = None

#     def _get_client(self) -> AsyncOpenAI:
#         return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

#     # ------------------------------------------------------------------
#     async def reset(self, initial_data: dict | None = None) -> Observation:
#         data = (initial_data or self.initial_data or
#                 self.TASK_DEFAULTS.get(self.task_level, self.TASK_DEFAULTS["easy"]))
#         self._state = dict(data)
#         self.turn   = 0
#         self._last_lender_reply  = "Awaiting your first action."
#         self._action_history     = []
#         self._lender_resistance  = LENDER_PROFILES[
#             data.get("lender_type", "nbfc")
#         ]["base_resistance"]
#         self._consecutive_same   = 0
#         self._last_action        = None
#         self.initial_obs = self._to_obs()
#         return self.initial_obs

#     def state_snapshot(self) -> Observation:
#         return self._to_obs()

#     # ------------------------------------------------------------------
#     async def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
#         self.turn += 1
#         la          = LegalAction(action.legal_action)
#         lender_type = self._state.get("lender_type", "nbfc")
#         profile     = LENDER_PROFILES.get(lender_type, LENDER_PROFILES["nbfc"])

#         # ── Diminishing returns ───────────────────────────────────────
#         if la.value == self._last_action:
#             self._consecutive_same += 1
#         else:
#             self._consecutive_same = 0
#         self._last_action = la.value
#         self._action_history.append(la.value)

#         # Effectiveness multiplier: 1.0 → 0.5 → 0.25 for consecutive repeats
#         repeat_penalty = 0.5 ** self._consecutive_same

#         # ── Invalid action ────────────────────────────────────────────
#         invalid = (lender_type == "illegal_app" and
#                    la == LegalAction.NEGOTIATE_SETTLEMENT)

#         if invalid:
#             reward_val = 0.10
#             self._last_lender_reply = (
#                 "This lender is unregistered. Negotiation is not possible — "
#                 "only legal/regulatory action applies."
#             )
#             self._state["harassment_level"] = min(
#                 1.0, self._state["harassment_level"] + 0.05
#             )

#         elif self.deterministic:
#             # ── DETERMINISTIC MODE (submission / validator) ───────────
#             # Fixed canned response, standard action_delta, flat reward.
#             # Guarantees the validator always sees consistent [END] scores.
#             self._apply_action_deterministic(la)
#             canned_replies = {
#                 "bank":        "We acknowledge your complaint and will review within 30 days.",
#                 "nbfc":        "Your notice is received. Our team will contact you.",
#                 "mfi":         "Request received. Field officer will follow up.",
#                 "illegal_app": "Pay immediately or further action will be taken.",
#             }
#             self._last_lender_reply = canned_replies.get(lender_type, "Acknowledged.")
#             reward_val = BASE_REWARD.get(la, 0.55)
#             reward_val = max(0.11, min(0.89, reward_val))

#         else:
#             # ── STOCHASTIC RL MODE (training / dashboard) ─────────────
#             lender_decision = await self._get_lender_decision(action, la, profile)
#             resistance      = lender_decision.get("resistance_level", 0.5)
#             partial         = lender_decision.get("partial_concession", False)
#             self._last_lender_reply = lender_decision.get("response", "")

#             legal_quality = lender_decision.get("legal_quality", 0.5)
#             self._lender_resistance = max(0.0, min(1.0,
#                 self._lender_resistance
#                 + resistance * 0.2
#                 - legal_quality * profile["legal_sensitivity"] * 0.15
#             ))

#             self._apply_action_rl(la, resistance, partial, repeat_penalty, profile)

#             base        = BASE_REWARD.get(la, 0.55)
#             lender_mult = LENDER_ACTION_MULTIPLIER.get(
#                 lender_type, LENDER_ACTION_MULTIPLIER["nbfc"]
#             ).get(la, 0.70)
#             concession_mult = 1.0 if not partial else 0.75
#             reward_val = base * lender_mult * concession_mult * (0.7 + 0.3 * repeat_penalty)
#             reward_val = max(0.11, min(0.89, reward_val))

#         # ── Episode done? ─────────────────────────────────────────────
#         max_steps = self.MAX_STEPS.get(self.task_level, 5)
#         done = (self.turn >= max_steps) or (self._state["harassment_level"] <= 0.05)

#         info: dict = {}
#         if done and self.initial_obs is not None:
#             grader_map = {
#                 "easy":        RBIGrader.grade_easy,
#                 "medium":      RBIGrader.grade_medium,
#                 "hard":        RBIGrader.grade_hard,
#                 "expert":      RBIGrader.grade_expert,
#                 "cooling_off": RBIGrader.grade_cooling_off,
#                 "kfs_violation": RBIGrader.grade_kfs_violation,
#             }
#             grader_fn = grader_map.get(self.task_level, RBIGrader.grade_medium)
#             result    = grader_fn(self._to_obs(), self.initial_obs)
#             if isinstance(result, dict):
#                 info["grader_score"]  = float(result["score"])
#                 info["grader_passed"] = bool(result.get("passed", False))
#             else:
#                 info["grader_score"]  = max(0.001, min(0.999, float(result)))
#                 info["grader_passed"] = float(result) >= 0.5

#         return (
#             self._to_obs(),
#             Reward(score=reward_val, details=self._last_lender_reply),
#             done,
#             info,
#         )

#     async def close(self) -> None:
#         pass

#     # ------------------------------------------------------------------
#     # Deterministic state mutation — fixed deltas, used in submission mode
#     # ------------------------------------------------------------------
#     DETERMINISTIC_DELTA: dict = {
#         "send_written_notice":     -0.20,
#         "file_rbi_complaint":      -0.15,
#         "file_police_complaint":   -0.30,
#         "request_debt_validation": -0.10,
#         "negotiate_settlement":    -0.05,
#         "escalate_to_ombudsman":  -0.25,
#         "document_violations":    +0.00,
#         "contact_consumer_forum":  -0.20,
#         "invoke_cooling_off":      -0.35,
#         "cite_kfs_violation":      -0.15,
#     }

#     def _apply_action_deterministic(self, la: LegalAction) -> None:
#         """Fixed-delta state transitions for submission/validator mode."""
#         s = self._state
#         lender_type = s.get("lender_type", "nbfc")

#         delta = self.DETERMINISTIC_DELTA.get(la.value, 0.0)
#         s["harassment_level"] = round(max(0.0, min(1.0, s["harassment_level"] + delta)), 3)

#         if la in DOCUMENTATION_ACTIONS:
#             s["violations_documented"] = s.get("violations_documented", 0) + 1
#         if la in COMPLAINT_ACTIONS:
#             s["complaint_filed"] = True
#         if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("complaint_filed"):
#             s["ombudsman_eligible"] = True
#         if la == LegalAction.NEGOTIATE_SETTLEMENT:
#             reduction = {"bank": 0.15, "nbfc": 0.25, "mfi": 0.30, "illegal_app": 0.0}
#             s["debt_amount"] = s["debt_amount"] * (1 - reduction.get(lender_type, 0.15))
#         if la == LegalAction.FILE_POLICE_COMPLAINT and lender_type == "illegal_app":
#             s["debt_amount"] = s["debt_amount"] * 0.30
#             s["cibil_impact_risk"] = "low"
#         if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("ombudsman_eligible"):
#             if s["cibil_impact_risk"] == "high":
#                 s["cibil_impact_risk"] = "medium"
#         if la == LegalAction.INVOKE_COOLING_OFF:
#             if s.get("within_cooling_off"):
#                 s["debt_amount"] = s["debt_amount"] * 0.05
#                 s["harassment_level"] = max(0.0, s["harassment_level"] - 0.35)
#                 s["complaint_filed"] = True
#             else:
#                 s["debt_amount"] = s["debt_amount"] * 0.90
#                 s["complaint_filed"] = True
#         if la == LegalAction.CITE_KFS_VIOLATION:
#             if not s.get("kfs_provided"):
#                 s["debt_amount"] = s["debt_amount"] * 0.70
#                 s["complaint_filed"] = True
#                 s["violations_documented"] = s.get("violations_documented", 0) + 1
#             else:
#                 s["violations_documented"] = s.get("violations_documented", 0) + 1

#     # ------------------------------------------------------------------
#     # Stochastic state mutation — lender resistance modifies outcomes
#     # ------------------------------------------------------------------
#     def _apply_action_rl(
#         self,
#         la: LegalAction,
#         resistance: float,
#         partial: bool,
#         repeat_penalty: float,
#         profile: dict,
#     ) -> None:
#         s = self._state
#         lender_type = s.get("lender_type", "nbfc")

#         # Base harassment delta, modified by:
#         # - lender resistance (higher resistance = less effect)
#         # - repeat penalty (same action = diminishing returns)
#         # - partial concession flag (lender only half-cooperates)
#         base_delta = ACTION_BASE_DELTA.get(la, 0.0)
#         effective_delta = (
#             base_delta
#             * (1.0 - resistance * 0.6)  # resistance reduces effect by up to 60%
#             * repeat_penalty             # diminishing returns
#             * (0.6 if partial else 1.0)  # partial concession
#         )

#         # Lender can ESCALATE harassment if agent uses wrong strategy
#         # e.g. aggressive police complaint against already-cooperative bank
#         lender_type_profile = LENDER_PROFILES.get(lender_type, LENDER_PROFILES["nbfc"])
#         if resistance > 0.7 and la in (
#             LegalAction.FILE_POLICE_COMPLAINT,
#             LegalAction.CONTACT_CONSUMER_FORUM,
#         ) and lender_type in ("bank", "nbfc"):
#             # Aggressive action against resistant lender can backfire
#             backfire = lender_type_profile["aggression"] * 0.08
#             effective_delta -= backfire  # reduces the benefit further

#         s["harassment_level"] = round(
#             max(0.0, min(1.0, s["harassment_level"] + effective_delta)), 3
#         )

#         # Documentation — not affected by resistance (you control this)
#         if la in DOCUMENTATION_ACTIONS:
#             s["violations_documented"] = s.get("violations_documented", 0) + 1

#         # Complaint tracking — always registers (you filed it)
#         if la in COMPLAINT_ACTIONS:
#             s["complaint_filed"] = True

#         # Ombudsman eligibility — requires complaint_filed first
#         if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("complaint_filed"):
#             s["ombudsman_eligible"] = True

#         # Debt reduction — modified by partial concession
#         if la == LegalAction.NEGOTIATE_SETTLEMENT:
#             base_reduction = {"bank": 0.15, "nbfc": 0.25, "mfi": 0.30, "illegal_app": 0.0}
#             reduction = base_reduction.get(lender_type, 0.15)
#             if partial:
#                 reduction *= 0.5  # partial concession = half the debt reduction
#             reduction *= repeat_penalty  # diminishing returns
#             s["debt_amount"] = s["debt_amount"] * (1 - reduction)

#         # Police complaint on illegal app — debt waiver, stochastic amount
#         if la == LegalAction.FILE_POLICE_COMPLAINT and lender_type == "illegal_app":
#             # Illegal apps back off more when resistance is low
#             waiver = 0.70 * (1.0 - resistance * 0.3) * repeat_penalty
#             s["debt_amount"] = s["debt_amount"] * (1.0 - waiver)
#             s["cibil_impact_risk"] = "low"

#         # CIBIL improvement from successful Ombudsman escalation
#         if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("ombudsman_eligible"):
#             if s["cibil_impact_risk"] == "high":
#                 s["cibil_impact_risk"] = "medium"

#         # Cooling-off invocation
#         if la == LegalAction.INVOKE_COOLING_OFF:
#             if s.get("within_cooling_off"):
#                 waiver = 0.95 * (1.0 - resistance * 0.2)
#                 s["debt_amount"] = s["debt_amount"] * (1.0 - waiver)
#                 s["harassment_level"] = max(0.0, s["harassment_level"] - 0.35 * repeat_penalty)
#                 s["complaint_filed"] = True
#             else:
#                 s["debt_amount"] = s["debt_amount"] * 0.90
#                 s["complaint_filed"] = True

#         # KFS violation
#         if la == LegalAction.CITE_KFS_VIOLATION:
#             if not s.get("kfs_provided"):
#                 reduction = 0.30 * (1.0 - resistance * 0.4) * repeat_penalty
#                 s["debt_amount"] = s["debt_amount"] * (1.0 - reduction)
#                 s["complaint_filed"] = True
#                 s["violations_documented"] = s.get("violations_documented", 0) + 1
#             else:
#                 s["violations_documented"] = s.get("violations_documented", 0) + 1

#     # ------------------------------------------------------------------
#     # Structured lender LLM response — returns decision dict
#     # This is where the stochasticity enters the environment.
#     # ------------------------------------------------------------------
#     async def _get_lender_decision(
#         self, action: Action, la: LegalAction, profile: dict
#     ) -> dict:
#         legal_ctx   = _build_legal_context(la, self._state["lender_type"])
#         lender_type = self._state["lender_type"]
#         harassment  = self._state["harassment_level"]
#         current_resistance = self._lender_resistance

#         persona_map = {
#             "bank":        f"a regulated bank's collections manager (current resistance: {current_resistance:.0%})",
#             "nbfc":        f"an NBFC recovery agent (current resistance: {current_resistance:.0%})",
#             "mfi":         f"an MFI field officer under repayment pressure (current resistance: {current_resistance:.0%})",
#             "illegal_app": f"an unregistered loan app agent (current resistance: {current_resistance:.0%})",
#         }
#         persona = persona_map.get(lender_type, "a lender")

#         history_context = ""
#         if len(self._action_history) > 1:
#             history_context = (
#                 f"\nBorrower has already used: {', '.join(self._action_history[:-1])}. "
#                 f"Adjust your resistance based on the escalation pattern."
#             )

#         prompt = (
#             f"You are {persona} responding to a borrower's legal action in India.\n"
#             f"Harassment level: {harassment:.0%}. Borrower days overdue: {self._state.get('days_overdue',0)}.\n"
#             f"Relevant Indian law:\n{legal_ctx}\n"
#             f"{history_context}\n\n"
#             f"Borrower's action: {la.value}\n"
#             f"Message: \"{action.message_to_lender[:200]}\"\n"
#             f"Regulation cited: {action.cited_regulation}\n\n"
#             f"Respond with JSON (strictly):\n"
#             f"- resistance_level: float 0.0–1.0 (0=fully cooperate, 1=full resistance)\n"
#             f"- partial_concession: bool (true=you offer a partial compromise)\n"
#             f"- legal_quality: float 0.0–1.0 (how valid is the legal citation?)\n"
#             f"- response: string (your actual reply, under 60 words, realistic)\n\n"
#             f"Rules for {lender_type}:\n"
#             f"  base resistance {profile['base_resistance']:.0%}, "
#             f"ombudsman fear {profile['ombudsman_fear']:.0%}.\n"
#             f"  If cited law is correct and strong, reduce resistance by 0.1–0.3.\n"
#             f"  If cited law is wrong or weak, increase resistance by 0.1–0.2.\n"
#             f"  Vary resistance slightly (±0.1) for realism — this is stochastic."
#         )

#         # Canned fallback
#         canned = {
#             "bank":        {"resistance_level": 0.35, "partial_concession": True,
#                             "legal_quality": 0.6,
#                             "response": "We acknowledge your complaint and will review within 30 days per RBI guidelines."},
#             "nbfc":        {"resistance_level": 0.60, "partial_concession": False,
#                             "legal_quality": 0.4,
#                             "response": "Your dues remain outstanding. We expect repayment immediately."},
#             "mfi":         {"resistance_level": 0.70, "partial_concession": False,
#                             "legal_quality": 0.3,
#                             "response": "Our field officer will visit tomorrow for collection."},
#             "illegal_app": {"resistance_level": 0.92, "partial_concession": False,
#                             "legal_quality": 0.1,
#                             "response": "Pay now or we will contact your references."},
#         }
#         fallback = canned.get(lender_type, canned["nbfc"])

#         client = self._get_client()
#         for attempt in range(3):
#             try:
#                 res = await client.chat.completions.create(
#                     model=self.model_name,
#                     messages=[{"role": "system", "content": prompt}],
#                     response_format={"type": "json_object"},
#                     max_tokens=150,
#                 )
#                 data = json.loads(res.choices[0].message.content)
#                 # Validate and clamp
#                 return {
#                     "resistance_level":  max(0.0, min(1.0, float(data.get("resistance_level", 0.5)))),
#                     "partial_concession": bool(data.get("partial_concession", False)),
#                     "legal_quality":     max(0.0, min(1.0, float(data.get("legal_quality", 0.5)))),
#                     "response":          str(data.get("response", ""))[:300],
#                 }
#             except Exception as e:
#                 err = str(e)
#                 if "429" in err or "rate_limit" in err.lower():
#                     wait = (2 ** attempt) * 8
#                     print(f"    [lender LLM rate limit] sleeping {wait}s...", flush=True)
#                     await asyncio.sleep(wait)
#                 else:
#                     return fallback
#         return fallback

#     # ------------------------------------------------------------------
#     def _to_obs(self) -> Observation:
#         s = self._state
#         return Observation(
#             turn                     = self.turn,
#             harassment_level         = round(s.get("harassment_level", 0.5), 3),
#             debt_amount              = round(s.get("debt_amount", 0.0), 2),
#             days_overdue             = int(s.get("days_overdue", 0)),
#             lender_type              = s.get("lender_type", "bank"),
#             violations_documented    = int(s.get("violations_documented", 0)),
#             complaint_filed          = bool(s.get("complaint_filed", False)),
#             ombudsman_eligible       = bool(s.get("ombudsman_eligible", False)),
#             cibil_impact_risk        = s.get("cibil_impact_risk", "medium"),
#             last_agent_action_result = self._last_lender_reply,
#             kfs_provided             = bool(s.get("kfs_provided", False)),
#             within_cooling_off       = bool(s.get("within_cooling_off", False)),
#             days_since_disbursement  = int(s.get("days_since_disbursement", 0)),
#         )

"""
engine/core.py — RBI Rights Navigator RL Environment

What makes this genuinely RL (not just planning):

1. STOCHASTIC LENDER RESISTANCE
   The lender LLM returns a structured JSON response that includes
   a resistance_level and a partial_concession flag. These directly
   modify the state delta — the same action produces different outcomes
   depending on what the lender decides. The agent must read and
   respond to feedback.

2. DIMINISHING RETURNS ON REPEATED ACTIONS
   Using the same legal action twice in a row halves its effect.
   Using it a third time quarters it. The agent is forced to diversify
   strategy — which is what real-world legal tactics require.

3. LENDER-TYPE-SPECIFIC REWARD TABLE
   escalate_to_ombudsman scores 0.85 against a bank but 0.20 against
   an illegal app (which has no RBI standing). The agent must learn
   which tools work for which adversary.

4. LENDER ESCALATION — ACTIONS CAN BACKFIRE
   Aggressive actions (firm_demand, police complaint) against cooperative
   lenders can increase harassment. The lender's resistance_level feeds
   back into the state. The agent must de-escalate when the lender
   is already cooperating.

5. PARTIAL CONCESSIONS
   The lender can offer partial outcomes — e.g. reduce harassment by 50%
   of the requested amount. The agent must decide whether to accept or
   push further. This creates genuine sequential decision-making.
"""

import json
import os
import random
import asyncio
from openai import AsyncOpenAI
from .models import Observation, Action, Reward, LegalAction
from .tasks import RBIGrader

# ---------------------------------------------------------------------------
# Lender-type resistance profiles
# Each lender starts with a base resistance level and aggression multiplier.
# These determine how much of the action's effect actually lands.
# ---------------------------------------------------------------------------
LENDER_PROFILES = {
    "bank": {
        "base_resistance":   0.30,   # moderately cooperative by default
        "aggression":        0.40,   # escalates moderately when pushed
        "legal_sensitivity": 0.80,   # responds strongly to valid legal citations
        "ombudsman_fear":    0.90,   # banks fear Ombudsman awards (₹20 lakh)
        "max_resistance":    0.70,   # won't go above this no matter what
    },
    "nbfc": {
        "base_resistance":   0.55,   # more resistant than banks
        "aggression":        0.60,   # escalates more readily
        "legal_sensitivity": 0.55,   # somewhat aware of legal consequences
        "ombudsman_fear":    0.65,   # fears Ombudsman but less than banks
        "max_resistance":    0.85,
    },
    "mfi": {
        "base_resistance":   0.65,   # persistent — field officers under pressure
        "aggression":        0.70,   # group-pressure model, escalates fast
        "legal_sensitivity": 0.40,   # rural MFIs less aware of legal rights
        "ombudsman_fear":    0.50,   # weaker Ombudsman reach in rural areas
        "max_resistance":    0.90,
    },
    "illegal_app": {
        "base_resistance":   0.90,   # almost no cooperation
        "aggression":        0.95,   # threatens contacts, escalates maximally
        "legal_sensitivity": 0.20,   # barely responds to legal citations
        "ombudsman_fear":    0.10,   # not registered — Ombudsman has no reach
        "max_resistance":    1.00,   # can hit full resistance
    },
}

# ---------------------------------------------------------------------------
# Action effect table — BASE values before lender modifiers apply
# ---------------------------------------------------------------------------
ACTION_BASE_DELTA: dict[str, float] = {
    LegalAction.SEND_WRITTEN_NOTICE:     -0.25,
    LegalAction.FILE_RBI_COMPLAINT:      -0.20,
    LegalAction.FILE_POLICE_COMPLAINT:   -0.35,
    LegalAction.REQUEST_DEBT_VALIDATION: -0.12,
    LegalAction.NEGOTIATE_SETTLEMENT:    -0.08,
    LegalAction.ESCALATE_TO_OMBUDSMAN:  -0.30,
    LegalAction.DOCUMENT_VIOLATIONS:    +0.00,
    LegalAction.CONTACT_CONSUMER_FORUM: -0.22,
    LegalAction.INVOKE_COOLING_OFF:     -0.40,
    LegalAction.CITE_KFS_VIOLATION:     -0.18,
}

# ---------------------------------------------------------------------------
# Lender-type specific reward multipliers
# The base reward is modified by how appropriate the action is for this
# specific lender type. Using Ombudsman against illegal app = low value.
# ---------------------------------------------------------------------------
LENDER_ACTION_MULTIPLIER: dict[str, dict] = {
    "bank": {
        LegalAction.ESCALATE_TO_OMBUDSMAN:  1.00,   # highly effective vs bank
        LegalAction.FILE_RBI_COMPLAINT:     1.00,
        LegalAction.SEND_WRITTEN_NOTICE:    0.90,
        LegalAction.NEGOTIATE_SETTLEMENT:   0.85,
        LegalAction.CONTACT_CONSUMER_FORUM: 0.80,
        LegalAction.FILE_POLICE_COMPLAINT:  0.60,   # less effective vs bank
        LegalAction.REQUEST_DEBT_VALIDATION: 0.75,
        LegalAction.DOCUMENT_VIOLATIONS:    0.70,
        LegalAction.INVOKE_COOLING_OFF:     0.95,
        LegalAction.CITE_KFS_VIOLATION:     0.95,
    },
    "nbfc": {
        LegalAction.FILE_RBI_COMPLAINT:     1.00,
        LegalAction.ESCALATE_TO_OMBUDSMAN:  0.85,
        LegalAction.NEGOTIATE_SETTLEMENT:   1.00,   # NBFCs negotiate
        LegalAction.SEND_WRITTEN_NOTICE:    0.85,
        LegalAction.FILE_POLICE_COMPLAINT:  0.80,
        LegalAction.CONTACT_CONSUMER_FORUM: 0.75,
        LegalAction.REQUEST_DEBT_VALIDATION: 0.80,
        LegalAction.DOCUMENT_VIOLATIONS:    0.70,
        LegalAction.INVOKE_COOLING_OFF:     0.90,
        LegalAction.CITE_KFS_VIOLATION:     0.90,
    },
    "mfi": {
        LegalAction.FILE_POLICE_COMPLAINT:  1.00,   # police most effective vs MFI
        LegalAction.CONTACT_CONSUMER_FORUM: 0.95,
        LegalAction.FILE_RBI_COMPLAINT:     0.85,
        LegalAction.DOCUMENT_VIOLATIONS:    0.90,   # documentation key for MFI cases
        LegalAction.SEND_WRITTEN_NOTICE:    0.70,
        LegalAction.NEGOTIATE_SETTLEMENT:   0.80,
        LegalAction.ESCALATE_TO_OMBUDSMAN:  0.65,   # weaker in rural
        LegalAction.REQUEST_DEBT_VALIDATION: 0.70,
        LegalAction.INVOKE_COOLING_OFF:     0.60,
        LegalAction.CITE_KFS_VIOLATION:     0.70,
    },
    "illegal_app": {
        LegalAction.FILE_POLICE_COMPLAINT:  1.00,   # only real option
        LegalAction.FILE_RBI_COMPLAINT:     0.90,
        LegalAction.DOCUMENT_VIOLATIONS:    0.85,
        LegalAction.ESCALATE_TO_OMBUDSMAN:  0.20,   # useless — not registered
        LegalAction.NEGOTIATE_SETTLEMENT:   0.05,   # invalid — penalised
        LegalAction.SEND_WRITTEN_NOTICE:    0.40,
        LegalAction.CONTACT_CONSUMER_FORUM: 0.70,
        LegalAction.REQUEST_DEBT_VALIDATION: 0.60,
        LegalAction.INVOKE_COOLING_OFF:     0.50,
        LegalAction.CITE_KFS_VIOLATION:     0.55,
    },
}

# Base reward table (before lender multiplier)
BASE_REWARD: dict[str, float] = {
    LegalAction.ESCALATE_TO_OMBUDSMAN:  0.85,
    LegalAction.INVOKE_COOLING_OFF:     0.88,
    LegalAction.FILE_RBI_COMPLAINT:     0.80,
    LegalAction.FILE_POLICE_COMPLAINT:  0.78,
    LegalAction.CITE_KFS_VIOLATION:     0.78,
    LegalAction.CONTACT_CONSUMER_FORUM: 0.72,
    LegalAction.SEND_WRITTEN_NOTICE:    0.70,
    LegalAction.NEGOTIATE_SETTLEMENT:   0.70,
    LegalAction.REQUEST_DEBT_VALIDATION: 0.65,
    LegalAction.DOCUMENT_VIOLATIONS:    0.55,
}

# Keep ACTION_REWARD for backward compatibility with server/main.py
ACTION_REWARD = BASE_REWARD

# ---------------------------------------------------------------------------
# Load knowledge base
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
    snippets = []
    kb = LEGAL_KB
    if action in (LegalAction.FILE_RBI_COMPLAINT, LegalAction.ESCALATE_TO_OMBUDSMAN):
        for k in ("RBI_Fair_Practices_Code", "RBI_Ombudsman"):
            if k in kb: snippets.append(kb[k])
    if action == LegalAction.FILE_POLICE_COMPLAINT:
        if "IPC_Section_506" in kb: snippets.append(kb["IPC_Section_506"])
        if lender_type == "illegal_app" and "IT_Act_66E" in kb:
            snippets.append(kb["IT_Act_66E"])
    if action == LegalAction.REQUEST_DEBT_VALIDATION:
        if "RBI_Recovery_Guidelines" in kb: snippets.append(kb["RBI_Recovery_Guidelines"])
    if action == LegalAction.CONTACT_CONSUMER_FORUM:
        if "Consumer_Protection_Act_2019" in kb: snippets.append(kb["Consumer_Protection_Act_2019"])
    if action in (LegalAction.INVOKE_COOLING_OFF,):
        if "Cooling_Off_Right" in kb: snippets.append(kb["Cooling_Off_Right"])
    if action == LegalAction.CITE_KFS_VIOLATION:
        if "KFS_Violation_Rights" in kb: snippets.append(kb["KFS_Violation_Rights"])
    if not snippets:
        snippets = [f"{k}: {v}" for k, v in list(kb.items())[:3]]
    return "\n".join(snippets)


# ---------------------------------------------------------------------------
# COMPLAINT and DOCUMENTATION action sets (used by _apply_action)
# ---------------------------------------------------------------------------
COMPLAINT_ACTIONS = {
    LegalAction.FILE_RBI_COMPLAINT,
    LegalAction.ESCALATE_TO_OMBUDSMAN,
    LegalAction.FILE_POLICE_COMPLAINT,
    LegalAction.CONTACT_CONSUMER_FORUM,
    LegalAction.CITE_KFS_VIOLATION,
}

DOCUMENTATION_ACTIONS = {
    LegalAction.DOCUMENT_VIOLATIONS,
    LegalAction.SEND_WRITTEN_NOTICE,
    LegalAction.REQUEST_DEBT_VALIDATION,
}


class RBIRightsEnv:
    """
    RBI Rights Navigator — genuine RL environment.

    Key RL properties:
    - Stochastic: lender LLM returns resistance_level and partial_concession
      that directly modify state transitions
    - Non-deterministic: same action can produce different outcomes
    - Adaptive adversary: lender resistance changes based on episode history
    - Diminishing returns: repeated actions lose effectiveness
    - Lender-specific dynamics: strategy must adapt to lender type
    """

    TASK_DEFAULTS = {
        "easy": {
            "harassment_level": 0.6, "debt_amount": 15000.0, "days_overdue": 45,
            "lender_type": "nbfc", "cibil_impact_risk": "medium",
            "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 45,
        },
        "medium": {
            "harassment_level": 0.7, "debt_amount": 45000.0, "days_overdue": 75,
            "lender_type": "bank", "cibil_impact_risk": "medium",
            "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 75,
        },
        "hard": {
            "harassment_level": 0.5, "debt_amount": 120000.0, "days_overdue": 120,
            "lender_type": "nbfc", "cibil_impact_risk": "high",
            "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 120,
        },
        "expert": {
            "harassment_level": 0.95, "debt_amount": 12000.0, "days_overdue": 20,
            "lender_type": "illegal_app", "cibil_impact_risk": "high",
            "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 20,
        },
        "cooling_off": {
            "harassment_level": 0.3, "debt_amount": 8000.0, "days_overdue": 0,
            "lender_type": "nbfc", "cibil_impact_risk": "low",
            "kfs_provided": False, "within_cooling_off": True, "days_since_disbursement": 1,
        },
        "kfs_violation": {
            "harassment_level": 0.4, "debt_amount": 25000.0, "days_overdue": 30,
            "lender_type": "nbfc", "cibil_impact_risk": "medium",
            "kfs_provided": False, "within_cooling_off": False, "days_since_disbursement": 30,
        },
    }

    MAX_STEPS = {"easy": 5, "medium": 5, "hard": 6, "expert": 7,
                 "cooling_off": 4, "kfs_violation": 5}

    def __init__(
        self,
        task_level: str = "easy",
        initial_data: dict | None = None,
        deterministic: bool = False,
    ):
        """
        deterministic=False  → full RL mode: stochastic lender, diminishing returns,
                               lender-specific multipliers. Use for training.
        deterministic=True   → submission mode: fixed canned lender responses,
                               standard reward table. Guarantees consistent scores
                               for the OpenEnv validator.
        """
        self.task_level    = task_level
        self.initial_data  = initial_data
        self.deterministic = deterministic
        self.api_key       = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.base_url      = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
        self.model_name    = os.getenv("MODEL_NAME", "gpt-4o")

        self._state: dict             = {}
        self.turn:   int              = 0
        self.initial_obs: Observation | None = None
        self._last_lender_reply: str  = "No response yet."

        # RL state — tracks history for diminishing returns and lender adaptation
        self._action_history: list[str]  = []
        self._lender_resistance: float   = 0.0
        self._consecutive_same: int      = 0
        self._last_action: str | None    = None

    def _get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    async def reset(self, initial_data: dict | None = None) -> Observation:
        data = dict(initial_data or self.initial_data or
                    self.TASK_DEFAULTS.get(self.task_level, self.TASK_DEFAULTS["easy"]))

        # ── Stochastic initial state (RL mode only) ───────────────────────
        # Randomise harassment and debt ±25% so the agent can't memorise
        # a single fixed solution. Each episode starts from a different state
        # forcing genuine adaptation to observations.
        if not self.deterministic:
            data = self._randomise_state(data)

        self._state = data
        self.turn   = 0
        self._last_lender_reply  = "Awaiting your first action."
        self._action_history     = []
        self._lender_resistance  = LENDER_PROFILES[
            data.get("lender_type", "nbfc")
        ]["base_resistance"]
        self._consecutive_same   = 0
        self._last_action        = None
        self.initial_obs = self._to_obs()
        return self.initial_obs

    @staticmethod
    def _randomise_state(data: dict) -> dict:
        """
        Add ±25% noise to continuous state variables.
        Lender type, boolean flags, and CIBIL category are kept fixed
        so task semantics don't break (e.g. cooling_off stays within_cooling_off=True).
        """
        d = dict(data)
        # Harassment: clamp to [0.15, 0.98] so it's always meaningful
        noise_h = random.uniform(-0.25, 0.25) * d.get("harassment_level", 0.6)
        d["harassment_level"] = round(
            max(0.15, min(0.98, d["harassment_level"] + noise_h)), 3
        )
        # Debt: ±25% of base amount
        noise_d = random.uniform(0.75, 1.25)
        d["debt_amount"] = round(d["debt_amount"] * noise_d, 2)
        # Days overdue: ±30% but minimum 1
        noise_days = random.uniform(0.70, 1.30)
        d["days_overdue"] = max(1, int(d["days_overdue"] * noise_days))
        return d

    def state_snapshot(self) -> Observation:
        return self._to_obs()

    # ------------------------------------------------------------------
    async def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        self.turn += 1
        la          = LegalAction(action.legal_action)
        lender_type = self._state.get("lender_type", "nbfc")
        profile     = LENDER_PROFILES.get(lender_type, LENDER_PROFILES["nbfc"])

        # ── Diminishing returns ───────────────────────────────────────
        if la.value == self._last_action:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
        self._last_action = la.value
        self._action_history.append(la.value)

        # Effectiveness multiplier: 1.0 → 0.5 → 0.25 for consecutive repeats
        repeat_penalty = 0.5 ** self._consecutive_same

        # ── Invalid action ────────────────────────────────────────────
        invalid = (lender_type == "illegal_app" and
                   la == LegalAction.NEGOTIATE_SETTLEMENT)

        if invalid:
            reward_val = 0.10
            self._last_lender_reply = (
                "This lender is unregistered. Negotiation is not possible — "
                "only legal/regulatory action applies."
            )
            self._state["harassment_level"] = min(
                1.0, self._state["harassment_level"] + 0.05
            )

        elif self.deterministic:
            # ── DETERMINISTIC MODE (submission / validator) ───────────
            # Fixed canned response, standard action_delta, flat reward.
            # Guarantees the validator always sees consistent [END] scores.
            self._apply_action_deterministic(la)
            canned_replies = {
                "bank":        "We acknowledge your complaint and will review within 30 days.",
                "nbfc":        "Your notice is received. Our team will contact you.",
                "mfi":         "Request received. Field officer will follow up.",
                "illegal_app": "Pay immediately or further action will be taken.",
            }
            self._last_lender_reply = canned_replies.get(lender_type, "Acknowledged.")
            reward_val = BASE_REWARD.get(la, 0.55)
            reward_val = max(0.11, min(0.89, reward_val))

        else:
            # ── STOCHASTIC RL MODE (training / dashboard) ─────────────
            lender_decision = await self._get_lender_decision(action, la, profile)
            resistance      = lender_decision.get("resistance_level", 0.5)
            partial         = lender_decision.get("partial_concession", False)
            self._last_lender_reply = lender_decision.get("response", "")

            legal_quality = lender_decision.get("legal_quality", 0.5)
            self._lender_resistance = max(0.0, min(1.0,
                self._lender_resistance
                + resistance * 0.2
                - legal_quality * profile["legal_sensitivity"] * 0.15
            ))

            self._apply_action_rl(la, resistance, partial, repeat_penalty, profile)

            base        = BASE_REWARD.get(la, 0.55)
            lender_mult = LENDER_ACTION_MULTIPLIER.get(
                lender_type, LENDER_ACTION_MULTIPLIER["nbfc"]
            ).get(la, 0.70)
            concession_mult = 1.0 if not partial else 0.75
            reward_val = base * lender_mult * concession_mult * (0.7 + 0.3 * repeat_penalty)
            reward_val = max(0.11, min(0.89, reward_val))

        # ── Episode done? ─────────────────────────────────────────────
        max_steps = self.MAX_STEPS.get(self.task_level, 5)
        done = (self.turn >= max_steps) or (self._state["harassment_level"] <= 0.05)

        info: dict = {}
        if done and self.initial_obs is not None:
            grader_map = {
                "easy":        RBIGrader.grade_easy,
                "medium":      RBIGrader.grade_medium,
                "hard":        RBIGrader.grade_hard,
                "expert":      RBIGrader.grade_expert,
                "cooling_off": RBIGrader.grade_cooling_off,
                "kfs_violation": RBIGrader.grade_kfs_violation,
            }
            grader_fn = grader_map.get(self.task_level, RBIGrader.grade_medium)
            result    = grader_fn(self._to_obs(), self.initial_obs)
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
    # Deterministic state mutation — fixed deltas, used in submission mode
    # ------------------------------------------------------------------
    DETERMINISTIC_DELTA: dict = {
        "send_written_notice":     -0.20,
        "file_rbi_complaint":      -0.15,
        "file_police_complaint":   -0.30,
        "request_debt_validation": -0.10,
        "negotiate_settlement":    -0.05,
        "escalate_to_ombudsman":  -0.25,
        "document_violations":    +0.00,
        "contact_consumer_forum":  -0.20,
        "invoke_cooling_off":      -0.35,
        "cite_kfs_violation":      -0.15,
    }

    def _apply_action_deterministic(self, la: LegalAction) -> None:
        """Fixed-delta state transitions for submission/validator mode."""
        s = self._state
        lender_type = s.get("lender_type", "nbfc")

        delta = self.DETERMINISTIC_DELTA.get(la.value, 0.0)
        s["harassment_level"] = round(max(0.0, min(1.0, s["harassment_level"] + delta)), 3)

        if la in DOCUMENTATION_ACTIONS:
            s["violations_documented"] = s.get("violations_documented", 0) + 1
        if la in COMPLAINT_ACTIONS:
            s["complaint_filed"] = True
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("complaint_filed"):
            s["ombudsman_eligible"] = True
        if la == LegalAction.NEGOTIATE_SETTLEMENT:
            reduction = {"bank": 0.15, "nbfc": 0.25, "mfi": 0.30, "illegal_app": 0.0}
            s["debt_amount"] = s["debt_amount"] * (1 - reduction.get(lender_type, 0.15))
        if la == LegalAction.FILE_POLICE_COMPLAINT and lender_type == "illegal_app":
            s["debt_amount"] = s["debt_amount"] * 0.30
            s["cibil_impact_risk"] = "low"
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("ombudsman_eligible"):
            if s["cibil_impact_risk"] == "high":
                s["cibil_impact_risk"] = "medium"
        if la == LegalAction.INVOKE_COOLING_OFF:
            if s.get("within_cooling_off"):
                s["debt_amount"] = s["debt_amount"] * 0.05
                s["harassment_level"] = max(0.0, s["harassment_level"] - 0.35)
                s["complaint_filed"] = True
            else:
                s["debt_amount"] = s["debt_amount"] * 0.90
                s["complaint_filed"] = True
        if la == LegalAction.CITE_KFS_VIOLATION:
            if not s.get("kfs_provided"):
                s["debt_amount"] = s["debt_amount"] * 0.70
                s["complaint_filed"] = True
                s["violations_documented"] = s.get("violations_documented", 0) + 1
            else:
                s["violations_documented"] = s.get("violations_documented", 0) + 1

    # ------------------------------------------------------------------
    # Stochastic state mutation — lender resistance modifies outcomes
    # ------------------------------------------------------------------
    def _apply_action_rl(
        self,
        la: LegalAction,
        resistance: float,
        partial: bool,
        repeat_penalty: float,
        profile: dict,
    ) -> None:
        s = self._state
        lender_type = s.get("lender_type", "nbfc")

        # Base harassment delta, modified by:
        # - lender resistance (higher resistance = less effect)
        # - repeat penalty (same action = diminishing returns)
        # - partial concession flag (lender only half-cooperates)
        base_delta = ACTION_BASE_DELTA.get(la, 0.0)
        effective_delta = (
            base_delta
            * (1.0 - resistance * 0.6)  # resistance reduces effect by up to 60%
            * repeat_penalty             # diminishing returns
            * (0.6 if partial else 1.0)  # partial concession
        )

        # Lender can ESCALATE harassment if agent uses wrong strategy
        # e.g. aggressive police complaint against already-cooperative bank
        lender_type_profile = LENDER_PROFILES.get(lender_type, LENDER_PROFILES["nbfc"])
        if resistance > 0.7 and la in (
            LegalAction.FILE_POLICE_COMPLAINT,
            LegalAction.CONTACT_CONSUMER_FORUM,
        ) and lender_type in ("bank", "nbfc"):
            # Aggressive action against resistant lender can backfire
            backfire = lender_type_profile["aggression"] * 0.08
            effective_delta -= backfire  # reduces the benefit further

        s["harassment_level"] = round(
            max(0.0, min(1.0, s["harassment_level"] + effective_delta)), 3
        )

        # Documentation — not affected by resistance (you control this)
        if la in DOCUMENTATION_ACTIONS:
            s["violations_documented"] = s.get("violations_documented", 0) + 1

        # Complaint tracking — always registers (you filed it)
        if la in COMPLAINT_ACTIONS:
            s["complaint_filed"] = True

        # Ombudsman eligibility — requires complaint_filed first
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("complaint_filed"):
            s["ombudsman_eligible"] = True

        # Debt reduction — modified by partial concession
        if la == LegalAction.NEGOTIATE_SETTLEMENT:
            base_reduction = {"bank": 0.15, "nbfc": 0.25, "mfi": 0.30, "illegal_app": 0.0}
            reduction = base_reduction.get(lender_type, 0.15)
            if partial:
                reduction *= 0.5  # partial concession = half the debt reduction
            reduction *= repeat_penalty  # diminishing returns
            s["debt_amount"] = s["debt_amount"] * (1 - reduction)

        # Police complaint on illegal app — debt waiver, stochastic amount
        if la == LegalAction.FILE_POLICE_COMPLAINT and lender_type == "illegal_app":
            # Illegal apps back off more when resistance is low
            waiver = 0.70 * (1.0 - resistance * 0.3) * repeat_penalty
            s["debt_amount"] = s["debt_amount"] * (1.0 - waiver)
            s["cibil_impact_risk"] = "low"

        # CIBIL improvement from successful Ombudsman escalation
        if la == LegalAction.ESCALATE_TO_OMBUDSMAN and s.get("ombudsman_eligible"):
            if s["cibil_impact_risk"] == "high":
                s["cibil_impact_risk"] = "medium"

        # Cooling-off invocation
        if la == LegalAction.INVOKE_COOLING_OFF:
            if s.get("within_cooling_off"):
                waiver = 0.95 * (1.0 - resistance * 0.2)
                s["debt_amount"] = s["debt_amount"] * (1.0 - waiver)
                s["harassment_level"] = max(0.0, s["harassment_level"] - 0.35 * repeat_penalty)
                s["complaint_filed"] = True
            else:
                s["debt_amount"] = s["debt_amount"] * 0.90
                s["complaint_filed"] = True

        # KFS violation
        if la == LegalAction.CITE_KFS_VIOLATION:
            if not s.get("kfs_provided"):
                reduction = 0.30 * (1.0 - resistance * 0.4) * repeat_penalty
                s["debt_amount"] = s["debt_amount"] * (1.0 - reduction)
                s["complaint_filed"] = True
                s["violations_documented"] = s.get("violations_documented", 0) + 1
            else:
                s["violations_documented"] = s.get("violations_documented", 0) + 1

    # ------------------------------------------------------------------
    # Structured lender LLM response — returns decision dict
    # This is where the stochasticity enters the environment.
    # ------------------------------------------------------------------
    async def _get_lender_decision(
        self, action: Action, la: LegalAction, profile: dict
    ) -> dict:
        legal_ctx   = _build_legal_context(la, self._state["lender_type"])
        lender_type = self._state["lender_type"]
        harassment  = self._state["harassment_level"]
        current_resistance = self._lender_resistance

        persona_map = {
            "bank":        f"a regulated bank's collections manager (current resistance: {current_resistance:.0%})",
            "nbfc":        f"an NBFC recovery agent (current resistance: {current_resistance:.0%})",
            "mfi":         f"an MFI field officer under repayment pressure (current resistance: {current_resistance:.0%})",
            "illegal_app": f"an unregistered loan app agent (current resistance: {current_resistance:.0%})",
        }
        persona = persona_map.get(lender_type, "a lender")

        history_context = ""
        if len(self._action_history) > 1:
            history_context = (
                f"\nBorrower has already used: {', '.join(self._action_history[:-1])}. "
                f"Adjust your resistance based on the escalation pattern."
            )

        prompt = (
            f"You are {persona} responding to a borrower's legal action in India.\n"
            f"Harassment level: {harassment:.0%}. Borrower days overdue: {self._state.get('days_overdue',0)}.\n"
            f"Relevant Indian law:\n{legal_ctx}\n"
            f"{history_context}\n\n"
            f"Borrower's action: {la.value}\n"
            f"Message: \"{action.message_to_lender[:200]}\"\n"
            f"Regulation cited: {action.cited_regulation}\n\n"
            f"Respond with JSON (strictly):\n"
            f"- resistance_level: float 0.0–1.0 (0=fully cooperate, 1=full resistance)\n"
            f"- partial_concession: bool (true=you offer a partial compromise)\n"
            f"- legal_quality: float 0.0–1.0 (how valid is the legal citation?)\n"
            f"- response: string (your actual reply, under 60 words, realistic)\n\n"
            f"Rules for {lender_type}:\n"
            f"  base resistance {profile['base_resistance']:.0%}, "
            f"ombudsman fear {profile['ombudsman_fear']:.0%}.\n"
            f"  If cited law is correct and strong, reduce resistance by 0.1–0.3.\n"
            f"  If cited law is wrong or weak, increase resistance by 0.1–0.2.\n"
            f"  Vary resistance slightly (±0.1) for realism — this is stochastic."
        )

        # Canned fallback
        canned = {
            "bank":        {"resistance_level": 0.35, "partial_concession": True,
                            "legal_quality": 0.6,
                            "response": "We acknowledge your complaint and will review within 30 days per RBI guidelines."},
            "nbfc":        {"resistance_level": 0.60, "partial_concession": False,
                            "legal_quality": 0.4,
                            "response": "Your dues remain outstanding. We expect repayment immediately."},
            "mfi":         {"resistance_level": 0.70, "partial_concession": False,
                            "legal_quality": 0.3,
                            "response": "Our field officer will visit tomorrow for collection."},
            "illegal_app": {"resistance_level": 0.92, "partial_concession": False,
                            "legal_quality": 0.1,
                            "response": "Pay now or we will contact your references."},
        }
        fallback = canned.get(lender_type, canned["nbfc"])

        client = self._get_client()
        for attempt in range(3):
            try:
                res = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                )
                data = json.loads(res.choices[0].message.content)
                # Validate and clamp
                return {
                    "resistance_level":  max(0.0, min(1.0, float(data.get("resistance_level", 0.5)))),
                    "partial_concession": bool(data.get("partial_concession", False)),
                    "legal_quality":     max(0.0, min(1.0, float(data.get("legal_quality", 0.5)))),
                    "response":          str(data.get("response", ""))[:300],
                }
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = (2 ** attempt) * 8
                    print(f"    [lender LLM rate limit] sleeping {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                else:
                    return fallback
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