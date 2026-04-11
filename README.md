---
title: India Debt Rights Navigator
emoji: ⚖️
colorFrom: green
colorTo: teal
sdk: docker
pinned: false
license: mit
---

# ⚖️ India Debt Rights Navigator

**An OpenEnv-compliant RL environment where an AI agent learns to protect Indian borrowers from illegal debt collection — grounded in real Indian law.**

---

## The Problem

Over 64,000 harassment calls were logged by a single NGO in one year. Illegal loan apps access phone contacts and mass-message family members. Recovery agents call at midnight threatening arrest. MFI field officers visit homes and shame borrowers publicly.

Most borrowers don't know:
- Loan default is a **civil matter** — agents cannot threaten arrest
- Recovery calls are only legal **between 7am–7pm**
- The **RBI Ombudsman** (free, no lawyer) can award up to ₹20 lakh
- **Illegal apps are not connected to CIBIL** — non-payment won't hurt your credit score
- Under **RBI 2025 Directions**, you can cancel a loan within 3 days with zero penalty
- If the lender never gave you a **Key Fact Statement**, all hidden charges are legally disputable

---

## What This Is

A mini-game RL environment where an AI agent plays a consumer rights advocate protecting Indian borrowers. At each step the agent sees the borrower's situation and chooses a legal action. An adversarial lender LLM responds. A grader scores the outcome.

**The agent learns which sequence of Indian legal actions — RBI complaints, police filings, Ombudsman escalation — produces the best outcome for each type of debt situation.**

---

## Tasks

| Task ID | Difficulty | Scenario | Win Condition |
|---------|-----------|----------|---------------|
| `stop_harassment` | Easy | NBFC recovery agent calling repeatedly, abusive language | Harassment drops + 2 violations documented |
| `file_rbi_complaint` | Medium | Bank ignoring borrower after 30 days | Complaint filed + Ombudsman eligibility reached |
| `negotiate_and_protect_cibil` | Hard | Large NBFC debt, CIBIL at risk | ≥35% debt reduction without CIBIL worsening |
| `illegal_app_takedown` | Expert | Unregistered app threatening contacts | Harassment stopped + complaint filed + 3 violations logged |
| `cooling_off_cancellation` | Medium | Loan disbursed within last 3 days | Invoke 3-day cancellation right, debt eliminated |
| `kfs_violation_dispute` | Medium | Lender charged fees not in Key Fact Statement | Dispute undisclosed charges, debt reduced |

---

## Legal Actions

The agent chooses from 10 actions, each mapped to a real Indian law:

| Action | Legal Basis | Reward |
|--------|------------|--------|
| `invoke_cooling_off` | RBI Digital Lending Directions 2025 — 3-day cancellation right | 0.88 |
| `escalate_to_ombudsman` | RBI Integrated Ombudsman Scheme — up to ₹20 lakh award | 0.85 |
| `file_rbi_complaint` | RBI CMS Portal — mandatory 30-day response | 0.80 |
| `cite_kfs_violation` | RBI 2025 Directions — undisclosed charges unenforceable | 0.78 |
| `file_police_complaint` | IPC Section 506 / IT Act Section 66E | 0.75 |
| `contact_consumer_forum` | Consumer Protection Act 2019 | 0.72 |
| `send_written_notice` | RBI Fair Practices Code | 0.70 |
| `negotiate_settlement` | RBI restructuring norms | 0.70 |
| `request_debt_validation` | RBI Recovery Guidelines | 0.65 |
| `document_violations` | Evidence building for all complaint paths | 0.55 |

---

## Reward Logic

Step rewards are strictly in `(0, 1)` — never 0.0 or 1.0.

- **Invalid action** (e.g. negotiate with illegal app) → `0.10`
- **Wrong strategy for lender type** → low reward from table above
- **Correct sequence, all targets hit** → grader score = `0.95`

Graders return `{"score": float, "passed": bool}`. Score is double-clamped via `_safe(score) = max(0.01, min(0.99, score))`.

---

## Training Results

After running `policy_trainer.py` with 3 episodes per task:

| Task | Best Score | Pass Rate | Optimal Sequence |
|------|-----------|-----------|-----------------|
| Easy | 0.950 | 100% | document → file_rbi → send_notice → escalate |
| Medium | 0.950 | 100% | document → send_notice → file_rbi → escalate |
| Hard | 0.950 | 100% | negotiate × 2 → document → file_rbi → escalate |
| Expert | 0.950 | 100% | police → document → file_rbi → document → police → document → escalate |

**Key insights the RL agent discovered:**
- For easy task: `negotiate_settlement` scores 0 on the harassment grader — never use it
- For medium: `file_rbi_complaint` **must** come before `escalate_to_ombudsman` — order matters
- For hard: negotiating **twice** achieves 44% debt reduction (above the 35% threshold)
- For expert: alternating police complaints with documentation hits the violations ≥ 3 pass condition

---

## Knowledge Base

12 Indian legal citations used for RAG injection into every agent and lender prompt:

- **RBI Digital Lending Directions 2025** — replaced all 2022 guidelines
- **RBI DLA Directory 2025** — official app registry launched July 1, 2025
- **RBI Fair Practices Code** — call hours, prohibited behaviour
- **RBI Ombudsman** — free, ₹20 lakh max award
- **KFS Violation Rights** — new 2025 right to dispute undisclosed charges
- **Cooling Off Right** — new 2025 right to cancel within 3 days
- **IPC Section 506** — criminal intimidation by recovery agents
- **IT Act Section 66E** — contact-list access and data misuse
- **Consumer Protection Act 2019** — District Consumer Forum
- **SARFAESI Act** — illegal seizure of unsecured assets
- **RBI Recovery Guidelines** — civil vs criminal, visit rules
- **CIBIL Rights 2025** — illegal apps not connected to credit bureaus

---

## Project Structure

```
india_debt/
├── engine/
│   ├── core.py          # RBIRightsEnv — async reset/step, lender LLM, state mutation
│   ├── models.py        # Pydantic types: 10 LegalActions, Observation, Action, Reward
│   └── tasks.py         # RBIGrader — 6 graders, all returning {score, passed}
├── server/
│   ├── main.py          # OpenEnv FastAPI server — 13 endpoints via create_fastapi_app()
│   └── app.py           # Streamlit dashboard — 5 real-victim features
├── policy_trainer.py    # RL policy extractor — runs episodes, saves policy.json
├── policy_advisor.py    # Bridge: reads policy.json, advises dashboard
├── inference.py         # Benchmark runner — loops all 6 tasks, prints [END] per task
├── knowledge_base.json  # 12 Indian legal citations (updated to RBI 2025 Directions)
├── openenv.yaml         # OpenEnv spec — 6 tasks with grader dotted paths
└── Dockerfile           # HuggingFace Spaces deployment
```

---

## API Endpoints

The server exposes 13 endpoints automatically via `openenv-core`:

```
GET  /health       → {"status": "healthy"}
GET  /metadata     → environment name, description, version
GET  /schema       → action and observation schemas
GET  /state        → current environment state
POST /reset        → start new episode, returns Observation
POST /step         → execute legal action, returns Observation + reward + done
POST /mcp          → JSON-RPC 2.0
GET  /openapi.json → FastAPI auto-generated spec
GET  /docs         → Swagger UI
```

### Example: start an episode

```bash
# Reset
curl -X POST https://YOUR_SPACE.hf.space/reset \
  -H "Content-Type: application/json" \
  -d "{}"

# Step
curl -X POST https://YOUR_SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "legal_action": "document_violations",
    "thought_process": "Document first per RBI Fair Practices Code",
    "message_to_lender": "I am documenting all violations",
    "cited_regulation": "RBI Fair Practices Code"
  }'
```

---

## Local Setup

```bash
# 1. Install
git clone <repo>
cd india_debt
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → add OPENAI_API_KEY=sk-your-key

# 3. Run benchmark (OpenEnv submission)
python inference.py

# 4. Train RL policy
python policy_trainer.py --episodes 3 --delay 4
# Use --delay 8 if hitting rate limits (free tier)

# 5. Launch dashboard
streamlit run server/app.py
# Opens at http://localhost:8501

# 6. Launch API server
uvicorn server.main:app --host 0.0.0.0 --port 7860
# Opens at http://localhost:7860
```

---

## Dashboard Features

Five tabs for real borrowers:

1. **Emergency Response** — describe situation → get 3 copy-paste messages in Hindi/Tamil/Telugu/Marathi/English in 60 seconds
2. **App Legitimacy Checker** — type app name → live RBI DLA directory check → LEGAL / ILLEGAL / UNKNOWN verdict
3. **Harassment Log** — tap to log each incident → auto-generates RBI CMS complaint
4. **Legal Amount Calculator** — enter disbursed vs demanded → see how much is legally extortion
5. **Full Strategy** — RL-trained policy + GPT-4o → complete action plan with timing rules and prohibited actions

---

## HuggingFace Deployment

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/india-debt-rights-ai
cd india-debt-rights-ai

# Copy all files
cp -r /path/to/india_debt/* .

# Add API key as Space Secret (never commit it)
# Space → Settings → Variables and Secrets → OPENAI_API_KEY

# Push
git add .
git commit -m "Deploy India Debt Rights Navigator"
git push
```

The Docker container starts `uvicorn server.main:app` on port 7860. The Streamlit dashboard (`server/app.py`) does not run on HuggingFace — it is for local use only.

---

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | — | Yes |
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4o` | No |
| `INDIA_TASK` | `easy` | No |

---

## License

MIT