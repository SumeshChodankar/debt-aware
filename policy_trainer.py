"""
policy_trainer.py — Runs multiple RL episodes per scenario, extracts
the best-performing action sequences, and saves them to policy.json.

This is the bridge between the RL environment and the real-user dashboard.

Usage:
    python policy_trainer.py              # runs 3 episodes per task
    python policy_trainer.py --episodes 10  # more episodes = better policy

What it produces:
    policy.json — a lookup table mapping situation profiles to:
      - best_sequence: ordered list of legal actions
      - best_score: grader score that sequence achieved
      - prohibited_actions: actions that consistently underperformed
      - timing_rules: when to skip steps based on situation
      - why: human-readable explanation of the strategy
"""

import asyncio
import os
import json
import argparse
import statistics
from collections import defaultdict
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from engine.core import RBIRightsEnv, LEGAL_KB, ACTION_REWARD
from engine.models import Action, LegalAction
from engine.tasks import RBIGrader

load_dotenv()

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o"

# ---------------------------------------------------------------------------
# System prompt with the 3 fixed rules learned from inference run analysis
# ---------------------------------------------------------------------------
_kb_text = "\n".join(f"- {k}: {v}" for k, v in LEGAL_KB.items())

def get_system_prompt(task: str) -> str:
    """Return a task-specific system prompt based on what the RL grader actually measures."""

    # Task-specific action guidance derived from grader analysis
    task_guidance = {
        "easy": (
            "TASK = stop_harassment. The grader ONLY measures harassment reduction + violations documented.\n"
            "NEVER use negotiate_settlement — it reduces debt (wrong metric) and barely touches harassment.\n"
            "OPTIMAL sequence: document_violations → file_rbi_complaint → send_written_notice → escalate_to_ombudsman\n"
            "Each step must be a DIFFERENT action."
        ),
        "medium": (
            "TASK = file_rbi_complaint. The grader measures complaint_filed (40%) + ombudsman_eligible (35%) + harassment (25%).\n"
            "CRITICAL: complaint_filed must be True BEFORE you use escalate_to_ombudsman.\n"
            "OPTIMAL: document_violations → send_written_notice → file_rbi_complaint → escalate_to_ombudsman\n"
            "Once complaint_filed=true in the observation, your NEXT action must be escalate_to_ombudsman."
        ),
        "hard": (
            "TASK = negotiate_settlement + protect CIBIL. Grader: debt_reduction (45%) + cibil (35%) + complaint (20%).\n"
            "USE negotiate_settlement TWICE — each use cuts remaining debt by 25% (NBFC rate).\n"
            "After negotiating twice, file_rbi_complaint then escalate_to_ombudsman to improve CIBIL.\n"
            "OPTIMAL: negotiate_settlement → negotiate_settlement → document_violations → file_rbi_complaint → escalate_to_ombudsman"
        ),
        "expert": (
            "TASK = illegal_app_takedown. Grader: harassment (30%) + legal complaints (30%) + debt (25%) + violations (15%).\n"
            "NEVER use negotiate_settlement — illegal apps have no legal standing.\n"
            "Pass condition: harassment ≤ 0.10 AND complaint_filed=true AND violations_documented ≥ 3\n"
            "CRITICAL ORDER: file_police_complaint → document_violations → file_rbi_complaint → document_violations → file_police_complaint → document_violations → escalate_to_ombudsman\n"
            "file_rbi_complaint must come BEFORE escalate_to_ombudsman — ombudsman needs complaint_filed=true first."
        ),
    }

    return f"""You are an expert Indian consumer rights advocate learning to protect borrowers.

Indian law you must cite:
{_kb_text}

{task_guidance.get(task, task_guidance["easy"])}

Respond with JSON:
- thought_process: cite specific law and explain why this action fits this task
- legal_action: choose the action that maximises THIS task's grader score
- message_to_lender: actual message text
- cited_regulation: specific RBI circular, IPC section, or Consumer Act clause"""


def get_action(client: OpenAI, obs_json: str, history: List[Dict], task: str = "easy") -> Action:
    """Single agent step with conversation history. Retries on rate limit with backoff."""
    messages = [{"role": "system", "content": get_system_prompt(task)}]
    # Keep only last 3 history turns to reduce token usage on long episodes
    for h in history[-3:]:
        messages.append({"role": "user",      "content": h["obs"]})
        messages.append({"role": "assistant", "content": h["action"]})
    messages.append({
        "role": "user",
        "content": f"Current situation:\n{obs_json}\n\nBest legal action now?"
    })

    for attempt in range(4):  # up to 4 attempts
        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=400,  # cap output tokens to stay under TPM
            )
            data = json.loads(res.choices[0].message.content)
            return Action(
                thought_process   = data.get("thought_process", ""),
                legal_action      = LegalAction(data.get("legal_action", "document_violations")),
                message_to_lender = data.get("message_to_lender", ""),
                cited_regulation  = data.get("cited_regulation", ""),
            )
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s
                print(f"    [rate limit] waiting {wait}s before retry {attempt+1}/4...", flush=True)
                import time; time.sleep(wait)
            else:
                break  # non-rate-limit error — use fallback immediately

    return Action(
        thought_process   = "Fallback after rate limit retries",
        legal_action      = LegalAction.DOCUMENT_VIOLATIONS,
        message_to_lender = "I am documenting all violations per RBI Fair Practices Code.",
        cited_regulation  = "RBI Fair Practices Code",
    )


async def run_episode(
    client: OpenAI,
    task: str,
    step_delay: float = 3.0,
) -> Dict[str, Any]:
    """
    Run one episode, return the action sequence + grader score.
    step_delay: seconds to sleep between steps to stay under TPM limit.
    """
    env     = RBIRightsEnv(task_level=task)
    obs     = await env.reset()
    history = []
    sequence = []
    rewards  = []
    score    = 0.0
    success  = False
    max_steps = env.MAX_STEPS.get(task, 5)

    for step in range(1, max_steps + 1):
        obs_json = obs.model_dump_json()
        action   = get_action(client, obs_json, history, task=task)

        history.append({
            "obs":    obs_json,
            "action": json.dumps({
                "legal_action":    action.legal_action.value,
                "cited_regulation": action.cited_regulation,
            })
        })

        # Inter-step sleep — prevents TPM exhaustion on longer episodes
        if step > 1 and step_delay > 0:
            await asyncio.sleep(step_delay)

        try:
            obs, reward_obj, done, info = await env.step(action)
        except Exception as e:
            # Rate limit or network error in lender LLM — use fallback state
            err = str(e)
            print(f"    [step {step} error] {err[:80]} — using fallback", flush=True)
            if "429" in err or "rate_limit" in err.lower():
                wait = 30
                print(f"    [rate limit] sleeping {wait}s before continuing...", flush=True)
                await asyncio.sleep(wait)
                try:
                    obs, reward_obj, done, info = await env.step(action)
                except Exception:
                    # Absolute fallback — return what we have so far
                    break
            else:
                break

        rewards.append(reward_obj.score)
        sequence.append(action.legal_action.value)

        if done:
            score   = info.get("grader_score", 0.0)
            success = info.get("grader_passed", False)
            break

    await env.close()
    return {
        "task":       task,
        "sequence":   sequence,
        "score":      score,
        "success":    success,
        "rewards":    rewards,
        "avg_reward": statistics.mean(rewards) if rewards else 0.0,
    }


def extract_policy(results_by_task: Dict[str, List[Dict]]) -> Dict:
    """
    From multiple episodes per task, extract:
    - best_sequence (highest grader score)
    - prohibited_actions (actions that always appeared in failed episodes)
    - timing_rules (derived from environment mechanics)
    - why (human-readable rationale)
    """
    policy = {}

    # Timing rules derived from environment mechanics (always true)
    TIMING_RULES = {
        "easy":   ["Document violations FIRST — harassment grader needs evidence before complaint is valid.",
                   "Do NOT use negotiate_settlement — grader measures harassment, not debt reduction.",
                   "file_rbi_complaint drops harassment -0.15, send_written_notice drops -0.20, escalate_to_ombudsman drops -0.25."],
        "medium": ["File complaint with lender BEFORE escalating — Ombudsman requires complaint_filed=True.",
                   "Use escalate_to_ombudsman immediately after complaint_filed becomes True in the observation.",
                   "Do not re-document after complaint is filed — every wasted step costs ombudsman eligibility."],
        "hard":   ["Use negotiate_settlement TWICE — each use reduces remaining debt by 25% (NBFC rate).",
                   "Cumulative reduction after 2x negotiate: 44% total — above the 35% needed to pass.",
                   "File complaint + escalate AFTER negotiating — escalate_to_ombudsman improves CIBIL from high to medium."],
        "expert": ["Never use negotiate_settlement — illegal apps have no RBI registration to negotiate against.",
                   "file_rbi_complaint MUST come before escalate_to_ombudsman — ombudsman needs complaint_filed=True.",
                   "Alternate file_police_complaint with document_violations — need violations_documented >= 3 to pass.",
                   "Optimal order: police → doc → file_rbi → doc → police → doc → escalate."],
    }

    # Derive prohibited actions from DATA: actions that appear ONLY in failed episodes
    # and NEVER in any passing episode for that task.
    # Do NOT use a static list — the static list contradicted the data.
    PROHIBITED = {}
    for task_name, episodes in results_by_task.items():
        passed_eps = [e for e in episodes if e["success"]]
        failed_eps = [e for e in episodes if not e["success"]]
        if failed_eps and passed_eps:
            passed_actions = set(a for e in passed_eps for a in e["sequence"])
            failed_only    = set(a for e in failed_eps for a in e["sequence"]) - passed_actions
            PROHIBITED[task_name] = list(failed_only)
        else:
            PROHIBITED[task_name] = []

    WHY = {
        "easy": (
            "For NBFC harassment at moderate severity: document first (builds evidence), "
            "file RBI complaint (triggers mandatory investigation), then written notice "
            "(formal paper trail). Harassment drops fastest when legal footing is established early."
        ),
        "medium": (
            "For bank ignoring borrower: the Ombudsman is the most powerful tool (₹20 lakh award) "
            "but requires a prior complaint + 30-day wait. Every step must build toward ombudsman_eligible=True. "
            "Re-documenting after complaint_filed wastes the step — escalate immediately."
        ),
        "hard": (
            "For large NBFC debt with CIBIL at risk: negotiate settlement TWICE before complaining "
            "(each pass cuts debt ~25%). Then file complaint and escalate to Ombudsman — "
            "escalation after complaint triggers CIBIL impact improvement from 'high' to 'medium'."
        ),
        "expert": (
            "For illegal apps: police complaint is the most effective single action "
            "(triggers 70% debt waiver in simulation). But the pass condition also requires "
            "violations_documented >= 3, so alternate police_complaint → document_violations → "
            "police_complaint. Never negotiate — illegal apps have no RBI standing."
        ),
    }

    for task, episodes in results_by_task.items():
        if not episodes:
            continue

        # Sort by grader score descending
        sorted_eps = sorted(episodes, key=lambda e: e["score"], reverse=True)
        best       = sorted_eps[0]
        all_scores = [e["score"] for e in episodes]
        passed     = [e for e in episodes if e["success"]]

        # Find actions that only appear in failed episodes
        failed = [e for e in episodes if not e["success"]]
        failed_actions = set()
        if failed and passed:
            passed_action_sets = [set(e["sequence"]) for e in passed]
            for fa in failed:
                for a in fa["sequence"]:
                    if all(a not in ps for ps in passed_action_sets):
                        failed_actions.add(a)

        policy[task] = {
            "best_sequence":      best["sequence"],
            "best_score":         round(best["score"], 4),
            "avg_score":          round(statistics.mean(all_scores), 4),
            "pass_rate":          round(len(passed) / len(episodes), 2),
            "episodes_run":       len(episodes),
            "prohibited_actions": list(set(PROHIBITED.get(task, [])) | failed_actions),
            "timing_rules":       TIMING_RULES.get(task, []),
            "why":                WHY.get(task, ""),
        }

    return policy


def build_situation_map(policy: Dict) -> Dict:
    """
    Build a situation classifier that maps real victim inputs
    to a task type, then to a policy.
    
    Situation dimensions:
      - lender_type:      bank / nbfc / mfi / illegal_app
      - harassment_level: low(<0.4) / medium(0.4-0.7) / high(>0.7)
      - days_overdue:     early(<30) / mid(30-90) / late(>90)
    """
    return {
        "situation_to_task": {
            # Illegal app → always expert track
            "illegal_app": "expert",

            # Bank / NBFC with high harassment → medium (complaint path)
            "bank_high_harassment":  "medium",
            "nbfc_high_harassment":  "easy",

            # Bank / NBFC with debt negotiation needed → hard
            "bank_large_debt":  "hard",
            "nbfc_large_debt":  "hard",

            # MFI → medium (complaint + documentation)
            "mfi": "medium",

            # Default → easy (stop harassment first)
            "default": "easy",
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
                "reason": "Early-stage severe harassment — stop it fast with documentation + notice."
            },
            {
                "condition": "debt_amount > 50000 and lender_type in ['bank','nbfc']",
                "task": "hard",
                "reason": "Large debt — settlement negotiation + CIBIL protection needed."
            },
            {
                "condition": "complaint_filed == False and days_overdue > 30",
                "task": "medium",
                "reason": "Overdue without complaint — Ombudsman path is highest leverage."
            },
            {
                "condition": "default",
                "task": "easy",
                "reason": "Start with harassment documentation and written notice."
            },
        ],
        "tasks": policy,
    }


async def main(n_episodes: int = 3, step_delay: float = 4.0) -> None:
    """
    n_episodes: episodes per task
    step_delay: seconds between each step within an episode.
                Default 4s keeps 5-step episode under 30K TPM at gpt-4o rates.
                Use --delay 8 on free tier (10K TPM limit).
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results_by_task: Dict[str, List[Dict]] = defaultdict(list)

    ALL_TASKS = ["easy", "medium", "hard", "expert"]

    print(f"\nRunning {n_episodes} episode(s) per task across {len(ALL_TASKS)} tasks...")
    print(f"Total episodes: {n_episodes * len(ALL_TASKS)}")
    print(f"Step delay: {step_delay}s  (increase with --delay N if hitting rate limits)\n")

    for task in ALL_TASKS:
        print(f"Task: {task}")
        for ep in range(n_episodes):
            result = await run_episode(client, task, step_delay=step_delay)
            results_by_task[task].append(result)
            status = "PASS" if result["success"] else "FAIL"
            print(f"  Episode {ep+1}: score={result['score']:.3f} [{status}] "
                  f"sequence={' → '.join(result['sequence'])}")

            # Inter-episode pause — longer than inter-step to let TPM window reset
            if ep < n_episodes - 1:
                episode_pause = max(step_delay * 2, 8.0)
                await asyncio.sleep(episode_pause)

        # Inter-task pause — give the token bucket time to refill
        print(f"  [pausing {step_delay * 3:.0f}s before next task...]")
        await asyncio.sleep(step_delay * 3)
        print()

    print("Extracting policy...")
    policy      = extract_policy(results_by_task)
    full_policy = build_situation_map(policy)

    out_path = os.path.join(os.path.dirname(__file__), "policy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_policy, f, indent=2, ensure_ascii=False)

    print(f"\nPolicy saved to {out_path}")
    print("\n=== Policy Summary ===")
    for task, data in policy.items():
        print(f"\n{task.upper()}")
        print(f"  Best sequence:  {' → '.join(data['best_sequence'])}")
        print(f"  Best score:     {data['best_score']}")
        print(f"  Avg score:      {data['avg_score']}")
        print(f"  Pass rate:      {data['pass_rate']*100:.0f}%")
        print(f"  Prohibited:     {data['prohibited_actions']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of RL episodes per task (default: 3)")
    parser.add_argument("--delay", type=float, default=4.0,
                        help=(
                            "Seconds to sleep between steps (default: 4). "
                            "Increase if hitting 429 rate limits. "
                            "TPM 30K limit → use 4. TPM 10K limit → use 8."
                        ))
    args = parser.parse_args()
    asyncio.run(main(n_episodes=args.episodes, step_delay=args.delay))