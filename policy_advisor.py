"""
policy_advisor.py — Reads the RL-trained policy.json and provides
situation-aware legal strategy to the dashboard.

This is the bridge:
  RL environment → policy_trainer.py → policy.json → policy_advisor.py → dashboard

The dashboard calls get_advice(situation) and receives a strategy grounded
in what the RL agent learned, not a raw GPT-4o guess.
"""

import os
import json
from typing import Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load policy — falls back to a built-in baseline if policy.json not yet run
# ---------------------------------------------------------------------------
_POLICY_PATH = os.path.join(os.path.dirname(__file__), "policy.json")

# Built-in baseline — derived from the inference run analysis.
# Covers the 4 task types. policy_trainer.py overwrites this with
# richer learned sequences once it has been run.
_BASELINE_POLICY = {
    "situation_to_task": {
        "illegal_app":          "expert",
        "bank_high_harassment": "medium",
        "nbfc_high_harassment": "easy",
        "bank_large_debt":      "hard",
        "nbfc_large_debt":      "hard",
        "mfi":                  "medium",
        "default":              "easy",
    },
    "classifier_rules": [
        {
            "condition": "lender_type == 'illegal_app'",
            "task": "expert",
            "reason": "Illegal apps require police + cybercrime complaints, not negotiation."
        },
        {
            "condition": "harassment_level > 0.7 and days_overdue < 30",
            "task": "easy",
            "reason": "Early severe harassment — stop it fast with documentation + notice."
        },
        {
            "condition": "debt_amount > 50000 and lender_type in ['bank','nbfc']",
            "task": "hard",
            "reason": "Large debt — settlement + CIBIL protection needed."
        },
        {
            "condition": "days_overdue > 60 and debt_amount < 50000",
            "task": "medium",
            "reason": "Overdue without large debt — file complaint and pursue Ombudsman path."
        },
        {
            "condition": "default",
            "task": "easy",
            "reason": "Start with harassment documentation and written notice."
        },
    ],
    "tasks": {
        "easy": {
            "best_sequence": [
                "document_violations",
                "file_rbi_complaint",
                "send_written_notice",
                "escalate_to_ombudsman",
            ],
            "best_score": 0.95,
            "prohibited_actions": ["negotiate_settlement"],
            "timing_rules": [
                "Document violations FIRST — harassment grader needs evidence before complaint is valid.",
                "Do NOT use negotiate_settlement — grader measures harassment not debt reduction.",
                "Stack harassment-reducing actions: file_rbi(-0.15) → notice(-0.20) → ombudsman(-0.25).",
            ],
            "why": (
                "grade_easy only measures harassment reduction + violations documented — not debt. "
                "Training showed negotiate_settlement wastes 2 steps on wrong metric, capping score at 0.71. "
                "Correct path: document → RBI complaint → written notice → Ombudsman escalation."
            ),
        },
        "medium": {
            "best_sequence": [
                "document_violations",
                "send_written_notice",
                "file_rbi_complaint",
                "escalate_to_ombudsman",
                "document_violations",
            ],
            "best_score": 0.95,
            "prohibited_actions": [],
            "timing_rules": [
                "file_rbi_complaint MUST come before escalate_to_ombudsman — ombudsman needs complaint_filed=True.",
                "Once complaint_filed=True in observation, next action MUST be escalate_to_ombudsman.",
                "Episodes 2,3,4 all hit 0.95 with this pattern. Episode 1 and 5 missed ombudsman step.",
            ],
            "why": (
                "Grader: complaint_filed (40%) + ombudsman_eligible (35%) + harassment (25%). "
                "Training confirmed: reaching both complaint_filed AND ombudsman_eligible = 0.95 score. "
                "Critical ordering: file complaint first, THEN escalate — not the other way around."
            ),
        },
        "hard": {
            "best_sequence": [
                "negotiate_settlement",
                "negotiate_settlement",
                "document_violations",
                "file_rbi_complaint",
                "escalate_to_ombudsman",
            ],
            "best_score": 0.95,
            "prohibited_actions": [],
            "timing_rules": [
                "Negotiate TWICE first — NBFC rate cuts remaining debt 25% each time (cumulative 44%).",
                "44% exceeds the 35% threshold needed to pass grade_hard.",
                "After two negotiations: document → file_rbi_complaint → escalate_to_ombudsman improves CIBIL.",
            ],
            "why": (
                "Training confirmed: 4 of 5 episodes hit 0.95 using negotiate x2 first. "
                "Debt: ₹1,20,000 → ₹90,000 → ₹67,500 (44% total reduction). "
                "Then complaint + Ombudsman escalation moves CIBIL from high to medium."
            ),
        },
        "expert": {
            "best_sequence": [
                "file_police_complaint",
                "document_violations",
                "file_rbi_complaint",
                "document_violations",
                "file_police_complaint",
                "document_violations",
                "escalate_to_ombudsman",
            ],
            "best_score": 0.896,
            "prohibited_actions": ["negotiate_settlement", "request_debt_validation"],
            "timing_rules": [
                "file_rbi_complaint MUST come before escalate_to_ombudsman — complaint_filed must be True first.",
                "Alternate file_police_complaint with document_violations — need violations_documented >= 3 to pass.",
                "Never use negotiate_settlement — illegal apps have no RBI standing.",
                "Training failure mode: escalating before complaint_filed=True scores ~0.87 but does not pass.",
            ],
            "why": (
                "Training: 3 of 5 passed, best score 0.896. Police complaint triggers 70%% debt waiver. "
                "Failures: episodes that escalated before filing RBI complaint missed the pass condition. "
                "Fixed order: police → doc → file_rbi → doc → police → doc → escalate."
            ),
        },
    },
}


def _load_policy() -> dict:
    """Load policy.json if it exists, otherwise return the built-in baseline."""
    try:
        with open(_POLICY_PATH, "r", encoding="utf-8") as f:
            p = json.load(f)
            # Validate it has the expected shape
            if "tasks" in p and "classifier_rules" in p:
                return p
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return _BASELINE_POLICY


POLICY = _load_policy()


# ---------------------------------------------------------------------------
# Situation classifier
# ---------------------------------------------------------------------------
def classify_situation(
    lender_type: str,
    harassment_level: float,
    days_overdue: int,
    debt_amount: float,
) -> tuple[str, str]:
    """
    Map real victim inputs to a task type using the RL-learned classifier rules.
    Returns (task_name, reason).
    """
    rules = POLICY.get("classifier_rules", [])

    for rule in rules:
        cond    = rule.get("condition", "default")
        task    = rule["task"]
        reason  = rule["reason"]

        if cond == "default":
            return task, reason
        try:
            if eval(cond, {}, {
                "lender_type":       lender_type,
                "harassment_level":  harassment_level,
                "days_overdue":      days_overdue,
                "debt_amount":       debt_amount,
                "complaint_filed":   False,
            }):
                return task, reason
        except Exception:
            continue

    return "easy", "Defaulting to harassment-stop strategy."


# ---------------------------------------------------------------------------
# Policy lookup
# ---------------------------------------------------------------------------
def get_policy_for_task(task: str) -> dict:
    """Return the full policy entry for a given task."""
    return POLICY.get("tasks", {}).get(task, _BASELINE_POLICY["tasks"]["easy"])


# ---------------------------------------------------------------------------
# Main advisor — called by the dashboard
# ---------------------------------------------------------------------------
def get_advice(
    lender_type: str,
    harassment_level: float,
    days_overdue: int,
    debt_amount: float,
    violation_type: str,
    client: Optional[OpenAI] = None,
    app_name: str = "",
    language: str = "English",
) -> dict:
    """
    Returns a complete strategy grounded in RL-learned policy.

    Structure:
      task             — which RL task this maps to
      task_reason      — why this task was chosen
      sequence         — ordered list of legal actions (from RL)
      step1_action     — the single best first action
      step1_message    — drafted message for step 1 (GPT-4o, focused)
      step1_regulation — cited regulation
      next_steps       — remaining sequence as human-readable list
      timing_rules     — RL-learned timing constraints
      prohibited       — actions the RL agent learned NOT to do
      why              — strategy rationale
      policy_source    — "rl_trained" or "baseline"
      score_expected   — best grader score this sequence achieved in simulation
    """
    # Step 1: classify situation using RL-learned rules
    task, task_reason = classify_situation(
        lender_type, harassment_level, days_overdue, debt_amount
    )

    # Step 2: look up RL policy for this task
    task_policy = get_policy_for_task(task)
    sequence    = task_policy.get("best_sequence", ["document_violations"])
    first_action = sequence[0] if sequence else "document_violations"

    # Step 3: use GPT-4o ONLY to draft the message for the first action
    # (not to decide the strategy — that already came from RL)
    step1_message    = ""
    step1_regulation = ""

    if client:
        from engine.core import LEGAL_KB
        kb_text = "\n".join(f"- {k}: {v}" for k, v in LEGAL_KB.items())

        lang_note = (
            f"Write the message in {language} script." if language != "English"
            else "Write the message in English."
        )

        prompt = (
            f"Indian consumer rights expert. A borrower is being harassed by "
            f"a {lender_type} lender (app: {app_name or 'unknown'}). "
            f"Violation: {violation_type}. Days overdue: {days_overdue}. "
            f"Harassment level: {harassment_level:.0%}.\n\n"
            f"The RL agent has determined the BEST first action is: {first_action}\n\n"
            f"Relevant law:\n{kb_text}\n\n"
            f"Draft ONLY the message for this specific action. {lang_note} "
            f"Return JSON: message (the text), cited_regulation (specific law), "
            f"key_fact (one thing the victim must know)."
        )
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Expert Indian consumer rights lawyer. Cite specific laws."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            data             = json.loads(res.choices[0].message.content)
            step1_message    = data.get("message", "")
            step1_regulation = data.get("cited_regulation", "")
            key_fact         = data.get("key_fact", "")
        except Exception as e:
            key_fact = ""
    else:
        key_fact = task_policy.get("why", "")

    # Step 4: format remaining sequence as human-readable next steps
    action_labels = {
        "send_written_notice":     "Send written notice to the lender's Nodal Officer",
        "file_rbi_complaint":      "File complaint on RBI CMS portal (cms.rbi.org.in)",
        "file_police_complaint":   "File police complaint — IPC Section 506 / IT Act 66E",
        "request_debt_validation": "Request written debt validation from lender",
        "negotiate_settlement":    "Negotiate a settlement offer with the lender",
        "escalate_to_ombudsman":   "Escalate to RBI Ombudsman — free, up to ₹20 lakh award",
        "document_violations":     "Document violations: save call logs, screenshots, timestamps",
        "contact_consumer_forum":  "File at District Consumer Forum under Consumer Protection Act 2019",
    }

    next_steps = [
        f"Step {i+2}: {action_labels.get(a, a)}"
        for i, a in enumerate(sequence[1:])
    ]

    policy_source = (
        "rl_trained"
        if os.path.exists(_POLICY_PATH)
        else "baseline"
    )

    return {
        "task":             task,
        "task_reason":      task_reason,
        "sequence":         sequence,
        "step1_action":     first_action,
        "step1_label":      action_labels.get(first_action, first_action),
        "step1_message":    step1_message,
        "step1_regulation": step1_regulation,
        "key_fact":         key_fact,
        "next_steps":       next_steps,
        "timing_rules":     task_policy.get("timing_rules", []),
        "prohibited":       task_policy.get("prohibited_actions", []),
        "why":              task_policy.get("why", ""),
        "policy_source":    policy_source,
        "score_expected":   task_policy.get("best_score", 0.0),
        "pass_rate":        task_policy.get("pass_rate", None),
        "episodes_run":     task_policy.get("episodes_run", None),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Policy Advisor — self test ===\n")
    print("Policy source:", "rl_trained" if os.path.exists(_POLICY_PATH) else "baseline (run policy_trainer.py to improve)")
    print()

    test_cases = [
        {"lender_type": "illegal_app", "harassment_level": 0.95, "days_overdue": 20,  "debt_amount": 12000,  "violation_type": "Morphed photos sent to contacts"},
        {"lender_type": "bank",        "harassment_level": 0.7,  "days_overdue": 75,  "debt_amount": 45000,  "violation_type": "Calling at midnight"},
        {"lender_type": "nbfc",        "harassment_level": 0.5,  "days_overdue": 120, "debt_amount": 120000, "violation_type": "Threatening arrest"},
        {"lender_type": "nbfc",        "harassment_level": 0.6,  "days_overdue": 45,  "debt_amount": 15000,  "violation_type": "Abusive language"},
    ]

    for tc in test_cases:
        result = get_advice(**tc)
        print(f"Situation: {tc['lender_type']} | harassment={tc['harassment_level']} | days={tc['days_overdue']} | debt=₹{tc['debt_amount']:,}")
        print(f"  → Task:      {result['task']} ({result['task_reason'][:60]}...)")
        print(f"  → Sequence:  {' → '.join(result['sequence'])}")
        print(f"  → First:     {result['step1_label']}")
        print(f"  → Prohibited: {result['prohibited']}")
        print(f"  → Score expected: {result['score_expected']}")
        print()
