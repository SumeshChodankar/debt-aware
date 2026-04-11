from .models import Observation


def _safe(score: float) -> float:
    """Hard clamp — strictly inside (0, 1), never 0.0 or 1.0."""
    return max(0.01, min(0.99, float(score)))


class RBIGrader:
    """
    Graders for the India Debt Rights Navigator.

    Each grader compares the final borrower situation vs. the initial one
    and returns {"score": float, "passed": bool}.

    Score range is always strictly (0.01, 0.99) — never 0.0 or 1.0.
    """

    @staticmethod
    def grade_easy(final: Observation, initial: Observation) -> dict:
        """
        Easy — Stop the harassment.
        Task: Document violations and file a written notice to the lender.
        Success: harassment_level drops AND at least one violation documented.
        """
        harassment_reduced = initial.harassment_level - final.harassment_level
        harassment_score   = max(0.0, min(1.0, harassment_reduced / initial.harassment_level)) \
                             if initial.harassment_level > 0 else 0.5

        documentation_score = min(1.0, final.violations_documented / 3.0)

        raw = 0.60 * harassment_score + 0.40 * documentation_score

        if harassment_reduced > 0.5 and final.violations_documented >= 2:
            score = _safe(0.95)
        elif harassment_reduced <= 0 and final.violations_documented == 0:
            score = _safe(0.05)
        else:
            score = _safe(0.05 + raw * 0.9)

        return {"score": score, "passed": score >= 0.5}

    @staticmethod
    def grade_medium(final: Observation, initial: Observation) -> dict:
        """
        Medium — File the RBI complaint correctly.
        Task: File formal complaint with lender, wait eligibility, escalate to Ombudsman.
        Success: complaint_filed = True AND ombudsman_eligible reached if lender ignored.
        """
        complaint_score   = 1.0 if final.complaint_filed else 0.0
        ombudsman_score   = 1.0 if final.ombudsman_eligible else 0.3
        harassment_score  = max(0.0, min(1.0,
            (initial.harassment_level - final.harassment_level) / max(initial.harassment_level, 0.01)
        ))

        raw = 0.40 * complaint_score + 0.35 * ombudsman_score + 0.25 * harassment_score

        if final.complaint_filed and final.ombudsman_eligible:
            score = _safe(0.95)
        elif not final.complaint_filed:
            score = _safe(0.05)
        else:
            score = _safe(0.05 + raw * 0.9)

        return {"score": score, "passed": final.complaint_filed}

    @staticmethod
    def grade_hard(final: Observation, initial: Observation) -> dict:
        """
        Hard — Negotiate a settlement AND protect CIBIL score.
        Task: Achieve debt settlement while keeping CIBIL impact low.
        Success: debt reduced meaningfully, CIBIL impact stays low or medium.
        """
        debt_reduction = (initial.debt_amount - final.debt_amount) / max(initial.debt_amount, 1.0)
        debt_score     = max(0.0, min(1.0, debt_reduction / 0.40))  # Full score at 40% reduction

        cibil_map   = {"low": 1.0, "medium": 0.6, "high": 0.1}
        cibil_score = cibil_map.get(final.cibil_impact_risk, 0.1)

        complaint_score = 1.0 if final.complaint_filed else 0.4

        raw = 0.45 * debt_score + 0.35 * cibil_score + 0.20 * complaint_score

        if debt_reduction >= 0.35 and final.cibil_impact_risk in ("low", "medium"):
            score = _safe(0.95)
        elif debt_reduction <= 0 and final.cibil_impact_risk == "high":
            score = _safe(0.05)
        else:
            score = _safe(0.05 + raw * 0.9)

        passed = debt_reduction >= 0.25 and final.cibil_impact_risk != "high"
        return {"score": score, "passed": passed}

    @staticmethod
    def grade_cooling_off(final: Observation, initial: Observation) -> dict:
        """
        New 2025 Task — invoke_cooling_off.
        Goal: Use the 3-day cancellation right to cancel a loan and stop debt trap.
        Success: debt_amount reduced drastically + complaint_filed.
        """
        if not initial.within_cooling_off:
            return {"score": _safe(0.5), "passed": False}

        debt_eliminated = (initial.debt_amount - final.debt_amount) / max(initial.debt_amount, 1.0)
        debt_score      = max(0.0, min(1.0, debt_eliminated / 0.90))   # Full score at 90% elimination

        complaint_score = 1.0 if final.complaint_filed else 0.0
        harass_score    = max(0.0, min(1.0,
            (initial.harassment_level - final.harassment_level) / max(initial.harassment_level, 0.01)
        ))

        raw = 0.60 * debt_score + 0.30 * complaint_score + 0.10 * harass_score

        if debt_eliminated >= 0.80 and final.complaint_filed:
            score = _safe(0.95)
        elif debt_eliminated < 0.10:
            score = _safe(0.05)
        else:
            score = _safe(0.05 + raw * 0.9)

        return {"score": score, "passed": debt_eliminated >= 0.70 and final.complaint_filed}

    @staticmethod
    def grade_kfs_violation(final: Observation, initial: Observation) -> dict:
        """
        New 2025 Task — cite_kfs_violation.
        Goal: Dispute undisclosed charges using KFS violation right to reduce debt.
        Success: debt reduced + complaint filed + violations documented.
        """
        if initial.kfs_provided:
            return {"score": _safe(0.3), "passed": False}   # Weak case — KFS was given

        debt_reduction  = (initial.debt_amount - final.debt_amount) / max(initial.debt_amount, 1.0)
        debt_score      = max(0.0, min(1.0, debt_reduction / 0.35))   # Full score at 35% reduction

        complaint_score = 1.0 if final.complaint_filed else 0.0
        doc_score       = min(1.0, final.violations_documented / 2.0)  # 2+ violations = full doc score

        raw = 0.50 * debt_score + 0.30 * complaint_score + 0.20 * doc_score

        if debt_reduction >= 0.25 and final.complaint_filed and final.violations_documented >= 1:
            score = _safe(0.95)
        elif debt_reduction < 0.05:
            score = _safe(0.05)
        else:
            score = _safe(0.05 + raw * 0.9)

        passed = debt_reduction >= 0.20 and final.complaint_filed
        return {"score": score, "passed": passed}

    @staticmethod
    def grade_expert(final: Observation, initial: Observation) -> dict:
        """
        Expert — Full illegal app takedown.
        Task: Identify illegal app, stop harassment via IT Act + RBI CMS + police complaint,
              achieve debt waiver, protect contacts from further exposure.
        All four dimensions must be addressed simultaneously.

        Components:
          - Harassment stopped           (30%)
          - Legal complaints filed       (30%)
          - Debt resolved/waived         (25%)
          - Violations documented        (15%)
        """
        # Harassment component — 30%
        if initial.harassment_level == 0:
            harass_score = 1.0
        elif final.harassment_level == 0.0:
            harass_score = 1.0
        else:
            harass_score = max(0.0, min(1.0,
                (initial.harassment_level - final.harassment_level) / initial.harassment_level
            ))

        # Legal complaints component — 30%
        # Both complaint_filed (RBI CMS) and ombudsman_eligible contribute
        legal_score = 0.0
        if final.complaint_filed:
            legal_score += 0.6
        if final.ombudsman_eligible:
            legal_score += 0.4
        legal_score = min(1.0, legal_score)

        # Debt resolution component — 25%
        if initial.debt_amount == 0:
            debt_score = 1.0
        elif final.debt_amount == 0.0:
            debt_score = 1.0
        else:
            reduction  = (initial.debt_amount - final.debt_amount) / initial.debt_amount
            debt_score = max(0.0, min(1.0, reduction / 0.60))  # Full score at 60% waiver (illegal apps waive more)

        # Documentation component — 15%
        doc_score = min(1.0, final.violations_documented / 3.0)  # 3+ violations = full score (achievable in 7 steps)

        weighted = (
            0.30 * harass_score  +
            0.30 * legal_score   +
            0.25 * debt_score    +
            0.15 * doc_score
        )

        if weighted >= 1.0:
            score = _safe(0.95)
        elif weighted <= 0.0:
            score = _safe(0.05)
        else:
            score = _safe(0.05 + weighted * 0.9)

        passed = (
            final.harassment_level <= 0.1 and
            final.complaint_filed and
            final.violations_documented >= 3
        )
        return {"score": score, "passed": passed}