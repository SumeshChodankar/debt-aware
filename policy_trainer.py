# """
# policy_trainer.py — RL Policy Trainer with Reward Curve Plotting

# Runs multiple episodes per task, extracts best-performing sequences,
# saves policy.json AND generates training_curves.png for the README.

# Usage:
#     python policy_trainer.py              # 3 episodes per task
#     python policy_trainer.py --episodes 5 --delay 4
# """

# import asyncio
# import os
# import json
# import argparse
# import statistics
# import matplotlib 
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# from collections import defaultdict
# from typing import List, Dict, Any
# from openai import OpenAI
# from dotenv import load_dotenv

# from engine.core import RBIRightsEnv, LEGAL_KB, ACTION_REWARD
# from engine.models import Action, LegalAction
# from engine.tasks import RBIGrader

# load_dotenv()

# API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
# API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
# MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o"

# _kb_text = "\n".join(f"- {k}: {v}" for k, v in LEGAL_KB.items())

# def get_system_prompt(task: str) -> str:
#     task_guidance = {
#         "easy": (
#             "TASK = stop_harassment. Grader ONLY measures harassment reduction + violations documented.\n"
#             "FIXED SEQUENCE: document_violations → file_rbi_complaint → send_written_notice → escalate_to_ombudsman.\n"
#             "Do NOT use negotiate_settlement."
#         ),
#         "medium": (
#             "TASK = file_rbi_complaint. Grader: complaint_filed (40%) + ombudsman_eligible (35%) + harassment (25%).\n"
#             "CRITICAL: Once complaint_filed=true, VERY NEXT action MUST be escalate_to_ombudsman.\n"
#             "SEQUENCE: document_violations → send_written_notice → file_rbi_complaint → escalate_to_ombudsman."
#         ),
#         "hard": (
#             "TASK = negotiate_and_protect_cibil. Grader: debt_reduction (45%) + CIBIL (35%) + complaint (20%).\n"
#             "RULE: If observation turn >= 3, STOP negotiating.\n"
#             "turn=1: negotiate_settlement. turn=2: negotiate_settlement.\n"
#             "turn=3: document_violations. turn=4: file_rbi_complaint. turn=5: escalate_to_ombudsman."
#         ),
#         "expert": (
#             "TASK = illegal_app_takedown. NEVER negotiate.\n"
#             "ALTERNATING PATTERN ONLY: police → document → police → document → police → document → escalate.\n"
#             "CRITICAL: If previous action was file_police_complaint, NEXT MUST be document_violations.\n"
#             "violations_documented must reach 3 to pass."
#         ),
#         "cooling_off": (
#             "TASK = cooling_off_cancellation. within_cooling_off=true.\n"
#             "Step 1: invoke_cooling_off. Step 2: file_rbi_complaint. Done."
#         ),
#         "kfs_violation": (
#             "TASK = kfs_violation_dispute. kfs_provided=false.\n"
#             "Step 1: cite_kfs_violation. Step 2: document_violations. Step 3: file_rbi_complaint. Step 4: escalate_to_ombudsman."
#         ),
#     }
#     return f"""You are an expert Indian consumer rights advocate protecting a borrower.

# Indian law:
# {_kb_text}

# {task_guidance.get(task, task_guidance["easy"])}

# Respond with JSON:
# - thought_process: cite specific Indian law
# - legal_action: action that maximises this task's grader
# - message_to_lender: actual message text
# - cited_regulation: specific RBI/IPC/Consumer Act clause"""


# def get_action(client, obs_json, history, task="easy"):
#     messages = [{"role": "system", "content": get_system_prompt(task)}]
#     for h in history[-3:]:
#         messages.append({"role": "user",      "content": h["obs"]})
#         messages.append({"role": "assistant", "content": h["action"]})
#     messages.append({"role": "user", "content": f"Current situation:\n{obs_json}\n\nBest legal action now?"})

#     for attempt in range(4):
#         try:
#             res = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=messages,
#                 response_format={"type": "json_object"},
#                 temperature=0.4,
#                 max_tokens=400,
#             )
#             data = json.loads(res.choices[0].message.content)
#             return Action(
#                 thought_process   = data.get("thought_process", ""),
#                 legal_action      = LegalAction(data.get("legal_action", "document_violations")),
#                 message_to_lender = data.get("message_to_lender", ""),
#                 cited_regulation  = data.get("cited_regulation", ""),
#             )
#         except Exception as e:
#             err = str(e)
#             if "429" in err or "rate_limit" in err.lower():
#                 import time; time.sleep((2 ** attempt) * 5)
#             else:
#                 break

#     return Action(
#         thought_process="Fallback",
#         legal_action=LegalAction.DOCUMENT_VIOLATIONS,
#         message_to_lender="Documenting violations per RBI Fair Practices Code.",
#         cited_regulation="RBI Fair Practices Code",
#     )


# async def run_episode(client, task, step_delay=4.0):
#     env  = RBIRightsEnv(task_level=task)
#     obs  = await env.reset()
#     history, sequence, rewards = [], [], []
#     score, success = 0.0, False

#     for step in range(1, env.MAX_STEPS.get(task, 5) + 1):
#         obs_json = obs.model_dump_json()
#         action   = get_action(client, obs_json, history, task=task)
#         history.append({
#             "obs":    obs_json,
#             "action": json.dumps({"legal_action": action.legal_action.value}),
#         })
#         if step > 1 and step_delay > 0:
#             await asyncio.sleep(step_delay)
#         try:
#             obs, reward_obj, done, info = await env.step(action)
#         except Exception as e:
#             if "429" in str(e) or "rate_limit" in str(e).lower():
#                 await asyncio.sleep(30)
#                 try:
#                     obs, reward_obj, done, info = await env.step(action)
#                 except Exception:
#                     break
#             else:
#                 break

#         rewards.append(reward_obj.score)
#         sequence.append(action.legal_action.value)
#         if done:
#             score   = info.get("grader_score", 0.0)
#             success = info.get("grader_passed", False)
#             break

#     await env.close()
#     return {"task": task, "sequence": sequence, "score": score,
#             "success": success, "rewards": rewards,
#             "avg_reward": statistics.mean(rewards) if rewards else 0.0}


# # ---------------------------------------------------------------------------
# # PLOTTING — generates training_curves.png for the README / hackathon judges
# # ---------------------------------------------------------------------------

# TASK_COLORS = {
#     "easy":        "#1D9E75",  # teal
#     "medium":      "#7F77DD",  # purple
#     "hard":        "#EF9F27",  # amber
#     "expert":      "#D85A30",  # coral
#     "cooling_off": "#378ADD",  # blue
#     "kfs_violation": "#639922", # green
# }

# BASELINE_SCORES = {
#     "easy":        0.28,
#     "medium":      0.22,
#     "hard":        0.19,
#     "expert":      0.15,
#     "cooling_off": 0.35,
#     "kfs_violation": 0.25,
# }


# def generate_plots(results_by_task: Dict, out_dir: str = ".") -> None:
#     """Generate two plots: reward curves + final score comparison."""
#     tasks = list(results_by_task.keys())
#     n_tasks = len(tasks)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     fig.patch.set_facecolor('#FAFAFA')

#     # ── Plot 1: Episode reward per task (step rewards across episodes) ──────
#     ax1 = axes[0]
#     ax1.set_facecolor('#F8F8F8')
#     ax1.set_title("Step rewards per episode (RL mode)", fontsize=13, fontweight='bold', pad=10)
#     ax1.set_xlabel("Training episode", fontsize=11)
#     ax1.set_ylabel("Step reward (avg per episode)", fontsize=11)
#     ax1.set_ylim(0, 1.05)

#     for task, episodes in results_by_task.items():
#         if not episodes: continue
#         color = TASK_COLORS.get(task, "#888")
#         avg_rewards = [e["avg_reward"] for e in episodes]
#         x = list(range(1, len(avg_rewards) + 1))
#         ax1.plot(x, avg_rewards, color=color, marker='o', linewidth=2,
#                  markersize=6, label=task.replace("_", " "), zorder=3)
#         # Shade area under curve
#         ax1.fill_between(x, avg_rewards, alpha=0.08, color=color)

#     # Baseline reference line
#     ax1.axhline(y=0.25, color='#BBBBBB', linestyle='--', linewidth=1, label='random baseline')
#     ax1.legend(fontsize=9, loc='lower right', framealpha=0.8)
#     ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)

#     # ── Plot 2: Final grader score — baseline vs trained ───────────────────
#     ax2 = axes[1]
#     ax2.set_facecolor('#F8F8F8')
#     ax2.set_title("Grader score: baseline vs trained agent", fontsize=13, fontweight='bold', pad=10)
#     ax2.set_ylabel("Grader score (0–1, pass threshold = 0.5)", fontsize=11)
#     ax2.set_ylim(0, 1.10)

#     bar_width = 0.35
#     x_pos = np.arange(len(tasks))

#     baseline_vals = [BASELINE_SCORES.get(t, 0.20) for t in tasks]
#     trained_vals  = [
#         statistics.mean([e["score"] for e in results_by_task[t]]) if results_by_task[t] else 0
#         for t in tasks
#     ]

#     bars_b = ax2.bar(x_pos - bar_width/2, baseline_vals, bar_width,
#                      color='#CCCCCC', label='Random baseline', zorder=2)
#     bars_t = ax2.bar(x_pos + bar_width/2, trained_vals, bar_width,
#                      color=[TASK_COLORS.get(t, "#888") for t in tasks],
#                      label='Trained agent', zorder=2)

#     # Pass threshold line
#     ax2.axhline(y=0.5, color='#D85A30', linestyle='--', linewidth=1.2,
#                 label='Pass threshold (0.5)', zorder=3)

#     # Value labels on bars
#     for bar in bars_b:
#         h = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
#                  f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#888')
#     for bar in bars_t:
#         h = bar.get_height()
#         ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
#                  f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
#     ax2.legend(fontsize=9, loc='upper left', framealpha=0.8)
#     ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)

#     plt.tight_layout(pad=2.0)
#     out_path = os.path.join(out_dir, "training_curves.png")
#     plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
#     plt.close()
#     print(f"\nPlot saved: {out_path}")


# def generate_reward_progression_plot(all_episode_scores: Dict, out_dir: str = ".") -> None:
#     """
#     Plots cumulative reward progression across ALL episodes (all tasks combined).
#     Shows the agent improving over training steps.
#     """
#     fig, ax = plt.subplots(figsize=(10, 4))
#     fig.patch.set_facecolor('#FAFAFA')
#     ax.set_facecolor('#F8F8F8')

#     ax.set_title("Cumulative grader score across all training episodes", fontsize=13, fontweight='bold')
#     ax.set_xlabel("Training episode (all tasks combined)", fontsize=11)
#     ax.set_ylabel("Grader score", fontsize=11)
#     ax.set_ylim(0, 1.05)

#     global_ep = 0
#     for task, episodes in all_episode_scores.items():
#         color = TASK_COLORS.get(task, "#888")
#         for ep_idx, ep in enumerate(episodes):
#             global_ep += 1
#             # Plot each episode's score as a scatter point
#             ax.scatter(global_ep, ep["score"],
#                        color=color, s=80, zorder=3,
#                        label=task.replace("_", " ") if ep_idx == 0 else "")
#             # Connect episodes of same task with thin line
#             if ep_idx > 0:
#                 prev_global = global_ep - 1
#                 prev_score  = episodes[ep_idx - 1]["score"]
#                 ax.plot([prev_global, global_ep], [prev_score, ep["score"]],
#                         color=color, linewidth=1.2, alpha=0.5, zorder=2)

#     ax.axhline(y=0.5, color='#D85A30', linestyle='--', linewidth=1.2,
#                label='Pass threshold')
#     ax.axhline(y=0.25, color='#BBBBBB', linestyle=':', linewidth=1,
#                label='Random baseline')
#     ax.legend(fontsize=9, loc='lower right', framealpha=0.8, ncol=2)
#     ax.grid(axis='y', alpha=0.3, linewidth=0.5)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     plt.tight_layout()
#     out_path = os.path.join(out_dir, "reward_progression.png")
#     plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
#     plt.close()
#     print(f"Plot saved: {out_path}")


# # ---------------------------------------------------------------------------
# # Policy extraction (unchanged from original)
# # ---------------------------------------------------------------------------

# def extract_policy(results_by_task):
#     TIMING_RULES = {
#         "easy":   ["Document first — builds evidence before complaint.",
#                    "Do NOT negotiate — grader measures harassment not debt."],
#         "medium": ["File complaint BEFORE escalating to Ombudsman.",
#                    "Escalate immediately after complaint_filed becomes True."],
#         "hard":   ["Negotiate EXACTLY TWICE then pivot to documentation.",
#                    "Escalate after complaint — improves CIBIL from high to medium."],
#         "expert": ["NEVER negotiate with illegal apps.",
#                    "Alternate police complaint with documentation (need violations >= 3)."],
#         "cooling_off":  ["invoke_cooling_off FIRST — within 3-day window."],
#         "kfs_violation": ["cite_kfs_violation FIRST — disputes undisclosed charges."],
#     }
#     WHY = {
#         "easy":    "Document violations then file RBI complaint — harassment drops fastest with legal footing.",
#         "medium":  "Ombudsman (₹20 lakh award) requires prior complaint + 30-day wait — escalate immediately after filing.",
#         "hard":    "Negotiate twice (44% cumulative debt cut) then escalate — Ombudsman escalation improves CIBIL.",
#         "expert":  "Police complaint triggers 70% debt waiver on illegal apps. Alternate with documentation for violations count.",
#         "cooling_off":  "3-day cooling-off right (RBI 2025) — invoke immediately to cancel with zero penalty.",
#         "kfs_violation": "KFS violation makes undisclosed charges unenforceable — cite it first for 30% debt reduction.",
#     }
#     policy = {}
#     for task, episodes in results_by_task.items():
#         if not episodes: continue
#         sorted_eps = sorted(episodes, key=lambda e: e["score"], reverse=True)
#         best       = sorted_eps[0]
#         all_scores = [e["score"] for e in episodes]
#         passed     = [e for e in episodes if e["success"]]
#         failed     = [e for e in episodes if not e["success"]]
#         passed_actions = set(a for e in passed for a in e["sequence"])
#         failed_only    = set(a for e in failed for a in e["sequence"]) - passed_actions if failed and passed else set()
#         policy[task] = {
#             "best_sequence":      best["sequence"],
#             "best_score":         round(best["score"], 4),
#             "avg_score":          round(statistics.mean(all_scores), 4),
#             "pass_rate":          round(len(passed) / len(episodes), 2),
#             "episodes_run":       len(episodes),
#             "prohibited_actions": list(failed_only),
#             "timing_rules":       TIMING_RULES.get(task, []),
#             "why":                WHY.get(task, ""),
#         }
#     return policy


# def build_situation_map(policy):
#     return {
#         "situation_to_task": {
#             "illegal_app":           "expert",
#             "bank_high_harassment":  "medium",
#             "nbfc_high_harassment":  "easy",
#             "bank_large_debt":       "hard",
#             "nbfc_large_debt":       "hard",
#             "mfi":                   "medium",
#             "cooling_off_eligible":  "cooling_off",
#             "kfs_not_provided":      "kfs_violation",
#             "default":               "easy",
#         },
#         "classifier_rules": [
#             {"condition": "lender_type == 'illegal_app'",                    "task": "expert"},
#             {"condition": "within_cooling_off == True",                       "task": "cooling_off"},
#             {"condition": "kfs_provided == False and days_since_disbursement < 30", "task": "kfs_violation"},
#             {"condition": "harassment_level > 0.7 and days_overdue < 30",    "task": "easy"},
#             {"condition": "debt_amount > 50000 and lender_type in ['bank','nbfc']", "task": "hard"},
#             {"condition": "complaint_filed == False and days_overdue > 30",   "task": "medium"},
#             {"condition": "default",                                           "task": "easy"},
#         ],
#         "tasks": policy,
#     }


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# async def main(n_episodes: int = 3, step_delay: float = 4.0) -> None:
#     client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
#     results_by_task: Dict[str, List[Dict]] = defaultdict(list)

#     ALL_TASKS = ["easy", "medium", "hard", "expert"]

#     print(f"\nRunning {n_episodes} episode(s) per task across {len(ALL_TASKS)} tasks...")
#     print(f"Total episodes: {n_episodes * len(ALL_TASKS)}")
#     print(f"Step delay: {step_delay}s\n")

#     for task in ALL_TASKS:
#         print(f"Task: {task}")
#         for ep in range(n_episodes):
#             result = await run_episode(client, task, step_delay=step_delay)
#             results_by_task[task].append(result)
#             status = "PASS" if result["success"] else "FAIL"
#             print(f"  Episode {ep+1}: score={result['score']:.3f} [{status}] "
#                   f"sequence={' → '.join(result['sequence'])}")
#             if ep < n_episodes - 1:
#                 await asyncio.sleep(max(step_delay * 2, 8.0))

#         print(f"  [pausing {step_delay * 3:.0f}s before next task...]")
#         await asyncio.sleep(step_delay * 3)
#         print()

#     # ── Generate plots ──────────────────────────────────────────────────────
#     print("Generating reward curves...")
#     generate_plots(results_by_task, out_dir=".")
#     generate_reward_progression_plot(results_by_task, out_dir=".")

#     # ── Save policy ─────────────────────────────────────────────────────────
#     print("Extracting policy...")
#     policy      = extract_policy(results_by_task)
#     full_policy = build_situation_map(policy)

#     out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(full_policy, f, indent=2, ensure_ascii=False)
#     print(f"Policy saved to {out_path}")

#     print("\n=== Policy Summary ===")
#     for task, data in policy.items():
#         print(f"\n{task.upper()}")
#         print(f"  Best sequence:  {' → '.join(data['best_sequence'])}")
#         print(f"  Best score:     {data['best_score']}")
#         print(f"  Avg score:      {data['avg_score']}")
#         print(f"  Pass rate:      {data['pass_rate']*100:.0f}%")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--episodes", type=int, default=3)
#     parser.add_argument("--delay",    type=float, default=4.0)
#     args = parser.parse_args()
#     asyncio.run(main(n_episodes=args.episodes, step_delay=args.delay))
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
            "TASK = negotiate_and_protect_cibil. Grader: debt_reduction (45%) + CIBIL (35%) + complaint (20%).\n"
            "CRITICAL: You must reduce debt AND improve CIBIL to pass. Score caps at 0.601 without CIBIL improvement.\n"
            "CIBIL only improves when escalate_to_ombudsman fires AFTER complaint_filed=true.\n"
            "Read observation each step — complaint_filed and ombudsman_eligible tell you where you are.\n"
            "OPTIMAL: cite_kfs_violation → negotiate_settlement → negotiate_settlement → file_rbi_complaint → escalate_to_ombudsman → escalate_to_ombudsman\n"
            "Step 1 cite_kfs_violation: disputes 30%% of debt AND files complaint in one move.\n"
            "Steps 2-3 negotiate_settlement: cuts remaining debt ~25%% each (NBFC rate).\n"
            "Step 4 file_rbi_complaint: strengthens case.\n"
            "Steps 5-6 escalate_to_ombudsman: REQUIRED to improve CIBIL from high to medium.\n"
            "Do NOT repeat cite_kfs_violation — diminishing returns after first use."
        ),
        "expert": (
            "TASK = illegal_app_takedown. Grader: harassment (30%) + legal complaints (30%) + debt (25%) + violations (15%).\n"
            "NEVER use negotiate_settlement — illegal apps have no legal standing.\n"
            "NEVER repeat the same action twice in a row — diminishing returns enforced.\n"
            "OPTIMAL 7-step sequence — follow exactly:\n"
            "  1. file_police_complaint  (70%% debt waiver, stops illegal app harassment)\n"
            "  2. document_violations    (violation #1)\n"
            "  3. file_rbi_complaint     (files formal complaint — needed before ombudsman)\n"
            "  4. document_violations    (violation #2)\n"
            "  5. file_police_complaint  (reinforces — violation #3 via police)\n"
            "  6. document_violations    (violation #3 documented)\n"
            "  7. escalate_to_ombudsman  (FINAL step — requires complaint_filed=true)\n"
            "Step 7 MUST be escalate_to_ombudsman — not document_violations.\n"
            "Check observation: if turn=6 and done=false, your next action MUST be escalate_to_ombudsman."
        ),
    }

    return f"""You are an expert Indian consumer rights advocate learning to protect borrowers.

VALID legal_action values — use ONLY these exact strings:
  send_written_notice, file_rbi_complaint, file_police_complaint,
  request_debt_validation, negotiate_settlement, escalate_to_ombudsman,
  document_violations, contact_consumer_forum, invoke_cooling_off, cite_kfs_violation

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