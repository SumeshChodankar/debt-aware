"""
India Debt Rights Navigator — OpenEnv-compliant HTTP server.

Mirrors DebtShield's proven structure exactly:
  create_fastapi_app() registers all required endpoints automatically.
"""

import sys
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Optional

import uvicorn
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action as OEAction,
    EnvironmentMetadata,
    Observation as OEObservation,
    State,
)
from pydantic import Field

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
from engine.core import RBIRightsEnv
from engine.models import Action, Observation, LegalAction

TASK_LEVEL = os.getenv("INDIA_TASK", "easy")


# ---------------------------------------------------------------------------
# OpenEnv-typed models
# ---------------------------------------------------------------------------

class RBIObservation(OEObservation):
    """Observation returned after each legal action step."""
    turn: int                    = Field(default=0)
    harassment_level: float      = Field(default=0.6)
    debt_amount: float           = Field(default=15000.0)
    days_overdue: int            = Field(default=45)
    lender_type: str             = Field(default="nbfc")
    violations_documented: int   = Field(default=0)
    complaint_filed: bool        = Field(default=False)
    ombudsman_eligible: bool     = Field(default=False)
    cibil_impact_risk: str       = Field(default="medium")
    last_agent_action_result: str = Field(default="Awaiting your first action.")


class RBIAction(OEAction):
    """Action submitted by the agent — a legal move to protect the borrower."""
    thought_process: str    = Field(default="", description="Agent reasoning citing RBI rule or IPC section")
    legal_action: str       = Field(
        default="document_violations",
        description=(
            "One of: send_written_notice, file_rbi_complaint, file_police_complaint, "
            "request_debt_validation, negotiate_settlement, escalate_to_ombudsman, "
            "document_violations, contact_consumer_forum"
        ),
    )
    message_to_lender: str  = Field(default="", description="Draft communication for the borrower to send")
    cited_regulation: str   = Field(default="", description="RBI circular, IPC section, or Consumer Act clause")


class RBIState(State):
    """Internal state of the RBI Rights Navigator environment."""
    harassment_level: float    = Field(default=0.6)
    debt_amount: float         = Field(default=15000.0)
    days_overdue: int          = Field(default=45)
    lender_type: str           = Field(default="nbfc")
    violations_documented: int = Field(default=0)
    complaint_filed: bool      = Field(default=False)
    ombudsman_eligible: bool   = Field(default=False)
    cibil_impact_risk: str     = Field(default="medium")


# ---------------------------------------------------------------------------
# OpenEnv Environment wrapper — bridges async core to sync interface
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run async coroutine safely regardless of whether a loop is running."""
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class IndiaDebtEnvironment(Environment[RBIAction, RBIObservation, RBIState]):
    """OpenEnv-compliant wrapper around RBIRightsEnv."""

    def __init__(self, task_level: str = TASK_LEVEL):
        super().__init__()
        self._task_level = task_level
        self._core = RBIRightsEnv(task_level=task_level)

    def reset(self, seed=None, episode_id=None, **kwargs) -> RBIObservation:
        obs: Observation = _run_async(self._core.reset())
        return self._to_oe(obs, done=False)

    def step(self, action: RBIAction, timeout_s=None, **kwargs) -> RBIObservation:
        try:
            la = LegalAction(action.legal_action)
        except ValueError:
            la = LegalAction.DOCUMENT_VIOLATIONS

        engine_action = Action(
            thought_process  = action.thought_process,
            legal_action     = la,
            message_to_lender = action.message_to_lender,
            cited_regulation = action.cited_regulation,
        )
        obs, reward_obj, done, info = _run_async(self._core.step(engine_action))
        oe_obs = self._to_oe(obs, done=done)
        oe_obs.reward = reward_obj.score
        if done and "grader_score" in info:
            oe_obs.metadata["grader_score"]  = info["grader_score"]
            oe_obs.metadata["grader_passed"] = info.get("grader_passed", False)
        return oe_obs

    @property
    def state(self) -> RBIState:
        s = self._core._state
        return RBIState(
            harassment_level    = s.get("harassment_level", 0.6),
            debt_amount         = s.get("debt_amount", 0.0),
            days_overdue        = int(s.get("days_overdue", 0)),
            lender_type         = s.get("lender_type", "bank"),
            violations_documented = int(s.get("violations_documented", 0)),
            complaint_filed     = bool(s.get("complaint_filed", False)),
            ombudsman_eligible  = bool(s.get("ombudsman_eligible", False)),
            cibil_impact_risk   = s.get("cibil_impact_risk", "medium"),
            step_count          = self._core.turn,
        )

    def close(self) -> None:
        try:
            _run_async(self._core.close())
        except Exception:
            pass

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name        = "India Debt Rights Navigator",
            description = (
                "A mini-game RL environment where an AI agent learns to protect Indian borrowers "
                "from illegal debt collection. The agent chooses legal actions (RBI complaints, "
                "police filings, Ombudsman escalation) against four lender types: bank, NBFC, "
                "MFI, and illegal app. Grounded in real Indian law: RBI Fair Practices Code, "
                "IPC Section 506, IT Act 66E, Consumer Protection Act 2019."
            ),
            version = "1.0.0",
            author  = "India Debt Rights AI",
        )

    @staticmethod
    def _to_oe(obs: Observation, done: bool) -> RBIObservation:
        return RBIObservation(
            turn                     = obs.turn,
            harassment_level         = obs.harassment_level,
            debt_amount              = obs.debt_amount,
            days_overdue             = obs.days_overdue,
            lender_type              = obs.lender_type,
            violations_documented    = obs.violations_documented,
            complaint_filed          = obs.complaint_filed,
            ombudsman_eligible       = obs.ombudsman_eligible,
            cibil_impact_risk        = obs.cibil_impact_risk,
            last_agent_action_result = obs.last_agent_action_result,
            done                     = done,
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env            = IndiaDebtEnvironment,
    action_cls     = RBIAction,
    observation_cls = RBIObservation,
)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
