import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from engine.core import RBIRightsEnv, LEGAL_KB
from engine.models import Action, LegalAction

load_dotenv()

# ---------------------------------------------------------------------------
# Mandatory OpenEnv variables
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o"
BENCHMARK    = "india_debt_rights"
TEMPERATURE  = 0.7
SUCCESS_SCORE_THRESHOLD = 0.1

ALL_TASKS = ["easy", "medium", "hard", "expert", "cooling_off", "kfs_violation"]
MAX_STEPS_PER_TASK = {"easy": 5, "medium": 5, "hard": 6, "expert": 7, "cooling_off": 4, "kfs_violation": 5}

# ---------------------------------------------------------------------------
# RAG-enhanced system prompt — Indian legal KB injected at startup
# ---------------------------------------------------------------------------
_kb_text = "\n".join(f"- {k}: {v}" for k, v in LEGAL_KB.items()) if LEGAL_KB else "No KB loaded."

SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are an expert Indian consumer rights advocate helping a borrower fight illegal debt collection.

    You know Indian law deeply:
    {_kb_text}

    At each step you must choose the most effective legal action for the situation and draft
    the actual communication the borrower should send.

    You must respond with a JSON object containing:
    - thought_process: your reasoning citing the specific RBI circular, IPC section, or law that applies
    - legal_action: one of [send_written_notice, file_rbi_complaint, file_police_complaint,
      request_debt_validation, negotiate_settlement, escalate_to_ombudsman,
      document_violations, contact_consumer_forum]
    - message_to_lender: the actual message the borrower should send (in formal but clear language)
    - cited_regulation: the specific regulation, section, or circular you are invoking

    Strategy guide:
    - For illegal_app lenders: NEVER negotiate. Use file_police_complaint and file_rbi_complaint.
    - For harassment: document_violations first, then send_written_notice, then escalate.
    - For debt relief: negotiate_settlement with banks/NBFCs after establishing legal footing.
    - Ombudsman requires complaint_filed first and 30 days elapsed.
    - Always cite a specific law — vague references are less effective.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging — strict OpenEnv format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = max(0.001, min(0.999, float(score)))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent — single LLM call
# ---------------------------------------------------------------------------
def get_model_action(client: OpenAI, obs_json: str) -> Action:
    user_prompt = (
        f"Current borrower situation:\n{obs_json}\n\n"
        f"What is the most effective legal action to take right now? "
        f"Cite the specific Indian law that applies."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            stream=False,
        )
        data = json.loads(completion.choices[0].message.content)
        return Action(
            thought_process   = data.get("thought_process", ""),
            legal_action      = LegalAction(data.get("legal_action", "document_violations")),
            message_to_lender = data.get("message_to_lender", ""),
            cited_regulation  = data.get("cited_regulation", ""),
        )
    except Exception as exc:
        return Action(
            thought_process   = f"Fallback: {exc}",
            legal_action      = LegalAction.DOCUMENT_VIOLATIONS,
            message_to_lender = "I am documenting all violations and will escalate to the RBI Ombudsman.",
            cited_regulation  = "RBI Fair Practices Code",
        )


# ---------------------------------------------------------------------------
# Single-task episode
# ---------------------------------------------------------------------------
async def run_task(client: OpenAI, task_name: str) -> None:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 5)

    env = RBIRightsEnv(task_level=task_name)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        for step in range(1, max_steps + 1):
            action = get_model_action(client, obs.model_dump_json())
            obs, reward_obj, done, info = await env.step(action)

            reward = reward_obj.score
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.model_dump_json(),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                score   = info.get("grader_score", 0.0)
                success = info.get("grader_passed", score >= SUCCESS_SCORE_THRESHOLD)
                break

    except Exception as e:
        print(f"[DEBUG] task={task_name} error={e}", flush=True)
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — runs ALL tasks, prints one [END] per task
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in ALL_TASKS:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
