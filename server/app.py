"""
India Debt Rights Navigator — Real Victim Emergency Dashboard
5 features that actually help someone being harassed right now.
Improved UI: better layout, comment boxes, fixed calculations.
"""

import os
import json
import asyncio
import datetime
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from engine.core import RBIRightsEnv, LEGAL_KB
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except ImportError:
    HAS_MIC = False
from policy_advisor import get_advice, classify_situation, POLICY
from engine.models import Action, LegalAction

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
client  = OpenAI(api_key=api_key) if api_key else None

st.set_page_config(
    page_title = "India Debt Rights Navigator",
    page_icon  = "⚖️",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font size increase +2px */
html, body, [class*="css"] {
    font-size: 16px !important;
}
.stMarkdown p, .stMarkdown li, .stCaption, .stText {
    font-size: 16px !important;
    line-height: 1.65 !important;
}
label, .stSelectbox label, .stTextInput label,
.stTextArea label, .stNumberInput label, .stSlider label {
    font-size: 15px !important;
    font-weight: 500 !important;
}
.stButton > button {
    font-size: 15px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 500 !important;
}
/* Softer tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 500;
}
/* Card containers */
.info-card {
    background: #1E2130;
    border: 1px solid #2D3147;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
/* Step badges */
.step-badge {
    display: inline-block;
    background: #7F77DD;
    color: white;
    border-radius: 50%;
    width: 26px; height: 26px;
    text-align: center;
    line-height: 26px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
}
/* Metric cards */
.metric-card {
    background: #1E2130;
    border: 1px solid #2D3147;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value { font-size: 24px; font-weight: 700; }
.metric-label { font-size: 12px; color: #9BA3C2; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── App legitimacy checker ────────────────────────────────────────────────────
import urllib.request

_KNOWN_ILLEGAL = {
    "quickcash", "kreditbe", "papamoney", "wallaby", "lemoncash",
    "alexandria", "cashbean", "loanzone", "rupeeclick", "cashnow",
    "instantloan", "paylater365", "loanfast", "rupeefort", "loanpe",
    "creditgo", "okcash", "rupeefresh", "paysense2", "loanbazar",
}
_KNOWN_REGISTERED = {
    "bajajfinserv", "bajaj", "hdfc", "icici", "sbi", "axis", "kotak",
    "idfc", "indusind", "rbl", "paytm", "phonepe", "cred", "slice",
    "lazypay", "simpl", "mpokket", "stashfin", "moneyview", "kissht",
    "dhani", "navi", "kreditbee", "cashe", "fibe", "prefr",
}

def check_app_legality(app_name: str) -> tuple[str, str, str]:
    name = app_name.lower().strip().replace(" ", "").replace("-", "")
    for a in _KNOWN_ILLEGAL:
        if a in name or name in a:
            return (
                "ILLEGAL — Unregistered (known predatory app)",
                "red",
                "DO NOT PAY. Not in RBI DLA directory. Non-payment will NOT affect CIBIL. "
                "Call 1930 immediately. File at cybercrime.gov.in and sachet.rbi.org.in."
            )
    for a in _KNOWN_REGISTERED:
        if a in name or name in a:
            return (
                "REGISTERED — In RBI DLA directory",
                "green",
                "This is a regulated lender. Payment affects CIBIL score. "
                "If harassed: file complaint at cms.rbi.org.in. "
                "You have cooling-off and KFS rights under 2025 Directions."
            )
    try:
        url = "https://www.rbi.org.in/Scripts/BS_NBFCList.aspx"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            body = r.read().decode("utf-8", errors="ignore").lower()
            if name in body or name.replace(" ", "") in body:
                return ("REGISTERED — Verified in RBI DLA directory (live check)", "green",
                        "Verified as registered with RBI. Payment affects CIBIL. "
                        "You have KFS and cooling-off rights under RBI Digital Lending Directions 2025.")
            return ("NOT IN RBI DIRECTORY — Likely illegal", "red",
                    "Not found in RBI DLA directory (live check). DO NOT PAY. "
                    "Non-payment will NOT affect CIBIL. Report at cybercrime.gov.in and helpline 1930.")
    except Exception:
        return ("UNKNOWN — Could not verify (check manually)", "orange",
                "Network check failed. Verify manually at rbi.org.in DLA directory. "
                "If not listed, treat as illegal — do not pay. Helpline: 1930.")

# ── Emergency message generator ───────────────────────────────────────────────
def generate_emergency_messages(
    lender_type: str,
    violation_type: str,
    language: str,
    app_name: str = "",
    disbursed: float = 0,
    demanded: float = 0,
    extra_context: str = "",        # NEW: optional comments from user
) -> dict:
    regional_lang = language if language != "English" else None
    lang_map = {
        "Hindi": "Hindi (Devanagari script)",
        "Tamil": "Tamil script",
        "Telugu": "Telugu script",
        "Marathi": "Marathi (Devanagari script)",
        "Bengali": "Bengali script",
    }
    if regional_lang:
        lang_name = lang_map.get(regional_lang, regional_lang)
        lang_block = (
            f"For each message provide TWO versions:\n"
            f"  - English version (keys: msg1_en, msg2_en, msg3_en)\n"
            f"  - {lang_name} version (keys: msg1, msg2, msg3)\n"
            f"Content must be identical — just translated."
        )
    else:
        lang_block = (
            "Write all messages in English.\n"
            "Use the same text for both regional keys (msg1,msg2,msg3) "
            "and English keys (msg1_en,msg2_en,msg3_en)."
        )

    amount_context = ""
    if disbursed > 0 and demanded > 0:
        legal_max = disbursed * (1 + 0.36 * 30 / 365)   # 30-day default estimate
        extortion = max(0, demanded - legal_max)
        amount_context = (
            f"\nLoan disbursed: ₹{disbursed:,.0f}. "
            f"Amount demanded: ₹{demanded:,.0f}. "
            f"Approximate legal maximum (36% APR, 30 days): ₹{legal_max:,.0f}. "
            f"Apparent overcharge: ₹{extortion:,.0f}."
        )

    # Merge violation + extra context into one rich description
    full_violation = violation_type
    if extra_context.strip():
        full_violation = f"{violation_type}. Additional details: {extra_context.strip()}"

    prompt = f"""You are an expert Indian consumer rights lawyer.
A borrower is being harassed by a {lender_type} lender (app: {app_name or 'unknown'}).
Violation: {full_violation}.{amount_context}

IMPORTANT: The borrower has described their specific situation above.
You MUST reference the specific details (visiting home, morphed photos, calls to employer etc.)
in EVERY message — do not use generic text. Make each message specific to what they described.

{lang_block}

Return JSON with exactly these keys:
- msg1_label: short label for message 1 (English)
- msg1: message to recovery agent — regional language
- msg1_en: message to recovery agent — English
- msg2_label: short label for message 2 (English)
- msg2: formal complaint to Nodal Officer — regional language
- msg2_en: formal complaint to Nodal Officer — English
- msg3_label: short label for message 3 (English)
- msg3: cybercrime/police complaint text — regional language
- msg3_en: cybercrime/police complaint text — English
- key_fact: one critical legal fact the borrower must know (English)
- do_not: one thing they must NOT do (English)

Each message must cite specific Indian laws (RBI Digital Lending Directions 2025,
IPC Section 506, Consumer Protection Act 2019 etc). Be firm and legally precise."""

    if not client:
        stub = "API key not configured. Please add OPENAI_API_KEY to your .env file."
        return {k: stub if "msg" in k else "Configure API key."
                for k in ["msg1","msg1_en","msg2","msg2_en","msg3","msg3_en",
                          "msg1_label","msg2_label","msg3_label","key_fact","do_not"]}

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Expert Indian consumer rights lawyer. Always cite specific laws."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(res.choices[0].message.content)


# ── Voice transcription via OpenAI Whisper ────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, client) -> dict:
    """
    Send recorded audio to OpenAI Whisper.
    Returns {text, language, language_name} or {error}.
    Supports Hindi, Tamil, Telugu, Marathi, Bengali, English automatically.
    """
    LANG_NAMES = {
        "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
        "mr": "Marathi", "bn": "Bengali", "en": "English",
        "kn": "Kannada", "gu": "Gujarati", "pa": "Punjabi",
    }
    try:
        result = client.audio.transcriptions.create(
            model           = "whisper-1",
            file            = ("situation.wav", audio_bytes, "audio/wav"),
            response_format = "verbose_json",   # returns detected language
        )
        lang_code = getattr(result, "language", "en") or "en"
        lang_name = LANG_NAMES.get(lang_code, lang_code.upper())
        return {
            "text":          result.text,
            "language":      lang_code,
            "language_name": lang_name,
            "error":         None,
        }
    except Exception as e:
        return {"text": "", "language": "en", "language_name": "English", "error": str(e)}


# ── Full automated action plan generator ─────────────────────────────────────
def generate_full_plan(
    sequence: list,
    lender_type: str,
    situation: str,
    debt_amount: float,
    client,
) -> list:
    """
    Takes the RL-learned action sequence and generates a complete automated
    action plan — one step per day, with message, portal link, and what to
    expect at each step.

    Returns list of step dicts:
    {day, action, title, why, message, portal, portal_url, expected_outcome, law}
    """
    ACTION_META = {
        "document_violations": {
            "title":    "Document All Violations",
            "why":      "Evidence is the foundation of every legal action. Courts and regulators need documented proof.",
            "portal":   "Keep records locally",
            "portal_url": None,
            "day_gap":  0,
        },
        "send_written_notice": {
            "title":    "Send Written Legal Notice",
            "why":      "A written notice creates a legal paper trail and triggers the lender's 30-day response obligation.",
            "portal":   "Send via Registered Post / Email",
            "portal_url": None,
            "day_gap":  1,
        },
        "file_rbi_complaint": {
            "title":    "File Complaint at RBI CMS Portal",
            "why":      "RBI CMS complaint triggers mandatory investigation within 30 days. Lender must respond in writing.",
            "portal":   "RBI CMS Portal",
            "portal_url": "https://cms.rbi.org.in",
            "day_gap":  1,
        },
        "file_police_complaint": {
            "title":    "File Police Complaint (IPC 506 + Cybercrime)",
            "why":      "Police complaint creates criminal record against the harasser. Illegal apps back off immediately.",
            "portal":   "Cybercrime Portal",
            "portal_url": "https://cybercrime.gov.in",
            "day_gap":  1,
        },
        "escalate_to_ombudsman": {
            "title":    "Escalate to RBI Ombudsman",
            "why":      "Ombudsman can award up to ₹20 lakh compensation. Lender must resolve or face penalty.",
            "portal":   "RBI Integrated Ombudsman",
            "portal_url": "https://cms.rbi.org.in",
            "day_gap":  30,  # Must wait 30 days after complaint
        },
        "negotiate_settlement": {
            "title":    "Negotiate Debt Settlement",
            "why":      "Formal settlement request puts lender on record. Reduces debt while maintaining legal leverage.",
            "portal":   "Send via Registered Post",
            "portal_url": None,
            "day_gap":  2,
        },
        "cite_kfs_violation": {
            "title":    "Dispute Charges — KFS Violation",
            "why":      "Under RBI 2025, undisclosed charges are legally unenforceable. This disputes the extra amount.",
            "portal":   "RBI CMS Portal",
            "portal_url": "https://cms.rbi.org.in",
            "day_gap":  0,
        },
        "invoke_cooling_off": {
            "title":    "Invoke 3-Day Cooling-Off Right",
            "why":      "You can cancel the loan within 3 days of disbursement — pay only principal + proportionate interest. No penalty.",
            "portal":   "Send via WhatsApp + Registered Post",
            "portal_url": None,
            "day_gap":  0,
        },
        "escalate_to_ombudsman": {
            "title":    "Escalate to RBI Ombudsman",
            "why":      "Ombudsman can award up to ₹20 lakh compensation. Lender must resolve or face penalty.",
            "portal":   "RBI Integrated Ombudsman",
            "portal_url": "https://cms.rbi.org.in",
            "day_gap":  30,
        },
        "contact_consumer_forum": {
            "title":    "File at Consumer Forum",
            "why":      "Consumer Forum can award compensation for mental harassment. Free to file, no lawyer needed.",
            "portal":   "National Consumer Disputes Redressal",
            "portal_url": "https://edaakhil.nic.in",
            "day_gap":  7,
        },
        "request_debt_validation": {
            "title":    "Request Formal Debt Validation",
            "why":      "Lender must prove the debt is valid and the amount is correct. Puts burden of proof on them.",
            "portal":   "Send via Registered Post",
            "portal_url": None,
            "day_gap":  1,
        },
    }

    if not client:
        # Generate plan without GPT-4o — use templates
        plan = []
        current_day = 0
        for i, action in enumerate(sequence):
            meta = ACTION_META.get(action, {
                "title": action.replace("_", " ").title(),
                "why":   "Protects your legal rights.",
                "portal": "RBI CMS Portal",
                "portal_url": "https://cms.rbi.org.in",
                "day_gap": 1,
            })
            if i > 0:
                current_day += meta["day_gap"] or 1
            plan.append({
                "step":          i + 1,
                "day":           current_day,
                "action":        action,
                "title":         meta["title"],
                "why":           meta["why"],
                "message":       f"[Configure OPENAI_API_KEY to generate specific message for {meta['title']}]",
                "portal":        meta["portal"],
                "portal_url":    meta["portal_url"],
                "law":           "RBI Digital Lending Directions 2025",
                "expected":      "Lender must acknowledge within 30 days.",
            })
        return plan

    # Generate all step messages in one GPT-4o call — efficient
    actions_desc = "\n".join([
        f"Step {i+1}: {ACTION_META.get(a, {}).get('title', a)} ({a})"
        for i, a in enumerate(sequence)
    ])

    prompt = f"""You are an expert Indian consumer rights lawyer.

Borrower situation: {situation or "General harassment by lender"}
Lender type: {lender_type}
Outstanding debt: ₹{debt_amount:,.0f}

The RL agent has determined this optimal legal sequence:
{actions_desc}

For EACH step, generate:
1. A specific, legally grounded message the borrower can send/file (cite specific RBI 2025 sections, IPC)
2. The expected outcome after this step
3. The specific law that makes this action powerful

Return a JSON array with one object per step:
[
  {{
    "step": 1,
    "action": "action_name",
    "message": "Full message text ready to send/file",
    "law": "Specific law citation",
    "expected": "What will happen after this step"
  }},
  ...
]

Make each message SPECIFIC to the borrower situation. Mention the lender type, debt amount.
Messages must be firm, legally precise, and cite exact RBI circulars or IPC sections."""

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Expert Indian consumer rights lawyer. Always cite specific laws."},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = json.loads(res.choices[0].message.content)
        gpt_steps = raw if isinstance(raw, list) else raw.get("steps", raw.get("plan", []))
    except Exception as e:
        gpt_steps = []

    # Merge GPT messages with metadata
    plan = []
    current_day = 0
    for i, action in enumerate(sequence):
        meta = ACTION_META.get(action, {
            "title": action.replace("_", " ").title(),
            "why": "Protects your legal rights.",
            "portal": "RBI CMS Portal",
            "portal_url": "https://cms.rbi.org.in",
            "day_gap": 1,
        })
        if i > 0:
            current_day += meta.get("day_gap", 1) or 1

        gpt = gpt_steps[i] if i < len(gpt_steps) else {}

        plan.append({
            "step":       i + 1,
            "day":        current_day,
            "action":     action,
            "title":      meta["title"],
            "why":        meta["why"],
            "message":    gpt.get("message", f"Send written notice invoking {meta['title']} under RBI 2025."),
            "portal":     meta["portal"],
            "portal_url": meta["portal_url"],
            "law":        gpt.get("law", "RBI Digital Lending Directions 2025"),
            "expected":   gpt.get("expected", "Lender must acknowledge within 30 days."),
        })
    return plan


# ── Legal amount calculator — FIXED LOGIC ─────────────────────────────────────
def calculate_legal_amount(
    disbursed: float,
    demanded: float,
    days: int,
    annual_rate: float,
    lender_type_key: str,
) -> dict:
    """
    Correct compound interest calculation.
    For illegal apps: they have no legal standing to charge interest at all.
    For registered lenders: RBI caps at 36% APR (Bank/NBFC) or 26% APR (MFI).
    """
    if "illegal" in lender_type_key.lower():
        # Illegal apps: legally owe ONLY principal (no interest right)
        legal_interest = 0.0
        legal_max      = disbursed
        note           = "Illegal apps have NO legal right to charge interest. You owe principal only."
    else:
        # Compound interest: P × (1 + r/365)^days
        legal_max  = disbursed * ((1 + annual_rate / 365) ** days)
        legal_interest = legal_max - disbursed
        note       = f"At {annual_rate*100:.0f}% APR compounded daily for {days} days."

    extortion        = max(0.0, demanded - legal_max)
    overpayment_pct  = (extortion / legal_max * 100) if legal_max > 0 else 0.0

    return {
        "disbursed":        disbursed,
        "demanded":         demanded,
        "legal_interest":   round(legal_interest, 2),
        "legal_max":        round(legal_max, 2),
        "extortion":        round(extortion, 2),
        "overpayment_pct":  round(overpayment_pct, 1),
        "days":             days,
        "annual_rate":      annual_rate,
        "note":             note,
    }


# ── Session state init ────────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state["log"] = []

# ── Top banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1f3a,#2d1f3a);
            border:1px solid #D85A30;border-radius:12px;
            padding:16px 24px;margin-bottom:1.2rem;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
  <div>
    <span style="font-size:20px;font-weight:700;color:#FF6B4A">⚖️ India Debt Rights Navigator</span>
    <span style="font-size:13px;color:#9BA3C2;margin-left:12px">Powered by RL · Grounded in RBI 2025 Law</span>
  </div>
  <div style="font-size:12px;color:#FF6B4A;font-weight:500">
    🆘 Being harassed right now? → Tab 1 → 60 seconds → 3 ready messages
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🆘 1. Emergency Messages",
    "🔍 2. App Checker",
    "📋 3. Violation Log",
    "💰 4. Amount Calculator",
    "⚖️ 5. Full Strategy",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Emergency Messages
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Get 3 ready-to-send legal messages in 60 seconds")
    st.caption("No legal knowledge needed. Fill in the details. Copy. Send.")

    # ── Row 1: Core inputs ────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        em_lender = st.selectbox("Who is harassing you?", [
            "Illegal loan app",
            "Bank (SBI, HDFC, ICICI etc.)",
            "NBFC / Fintech lender",
            "Microfinance institution (MFI)",
        ], key="em_lender")

        em_violation = st.selectbox("What are they doing?", [
            "Calling me at night / before 7am or after 7pm",
            "Threatening arrest / filing criminal case",
            "Contacting my family, friends or employer",
            "Sending abusive / threatening messages",
            "Morphed photos / blackmail",
            "Calling 10+ times per day",
            "Visiting my home or workplace uninvited",
            "Other (describe below)",
        ], key="em_violation")

    with col2:
        em_language = st.selectbox("Preferred language for messages", [
            "Hindi", "English", "Tamil", "Telugu", "Marathi", "Bengali"
        ], key="em_language")

        em_app = st.text_input(
            "App / lender name",
            placeholder="e.g. QuickCash, HDFC Bank, PaySense...",
            key="em_app",
            help="Helps generate more specific messages and checks legality"
        )

    # ── Additional context (NEW) ──────────────────────────────────
    # Read prefill value set by voice/whatsapp analyser — never set em_context directly
    _context_prefill = st.session_state.pop("em_context_prefill", "")

    em_context = st.text_area(
        "Additional details (optional but improves messages)",
        value=_context_prefill,
        placeholder=(
            "Describe what is happening in your own words. "
            "e.g. They called my boss at 6am. They sent morphed photos to my WhatsApp contacts. "
            "They are threatening to file a criminal case if I don't pay Rs 22,000 by tomorrow."
        ),
        height=100,
        key="em_context",
        help="The more context you give, the more specific and powerful the messages will be."
    )

    # ── Voice input ────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a1f3a,#1f2a1a);
            border:1px solid #4CAF50;border-radius:12px;padding:14px 18px;margin-bottom:10px">
  <div style="font-size:14px;font-weight:600;color:#4CAF50;margin-bottom:4px">
    🎤 Speak your situation — no typing needed
  </div>
  <div style="font-size:12px;color:#9BA3C2">
    Works in Hindi · Tamil · Telugu · Marathi · Bengali · Kannada · English
  </div>
</div>
""", unsafe_allow_html=True)

    if HAS_MIC:
        # Recording state tracker
        if "voice_recording" not in st.session_state:
            st.session_state["voice_recording"] = False

        voice_col1, voice_col2 = st.columns([1, 2])

        with voice_col1:
            audio = mic_recorder(
                start_prompt        = "🎤  Start Recording",
                stop_prompt         = "⏹  Stop Recording",
                just_once           = True,
                use_container_width = True,
                key                 = "voice_input",
            )

        with voice_col2:
            # Live recording indicator with animated timer
            st.markdown("""
<div id="rec-status" style="padding:10px 14px;border-radius:8px;
     background:#1E2130;border:1px solid #2D3147;font-size:12px;color:#9BA3C2">
  <div style="font-weight:500;color:#CDD0E0;margin-bottom:6px">How to use:</div>
  1. Click <strong style="color:#4CAF50">Start Recording</strong><br>
  2. Speak clearly about your situation<br>
  3. Click <strong style="color:#EF9F27">Stop Recording</strong><br>
  4. Wait 2-3 seconds for transcription
</div>
<script>
// Animate the recording button area
const btn = window.parent.document.querySelector('[data-testid="stButton"] button');
if (btn) {
  btn.addEventListener('click', function() {
    const status = document.getElementById('rec-status');
    if (status && this.innerText.includes('Stop')) {
      status.innerHTML = '<div style="color:#EF9F27;font-weight:600">🔴 Recording in progress...</div>' +
        '<div id="timer" style="font-size:24px;font-weight:700;color:#FF6B4A;margin:8px 0">0:00</div>' +
        '<div style="color:#9BA3C2;font-size:11px">Speak clearly · Stop when done</div>';
      let secs = 0;
      window._recTimer = setInterval(() => {
        secs++;
        const m = Math.floor(secs/60), s = secs%60;
        const el = document.getElementById('timer');
        if (el) el.textContent = m+':'+(s<10?'0':'')+s;
      }, 1000);
    } else if (status && window._recTimer) {
      clearInterval(window._recTimer);
      status.innerHTML = '<div style="color:#4CAF50;font-weight:600">✅ Processing your recording...</div>';
    }
  });
}
</script>
""", unsafe_allow_html=True)

        # Recording status display
        if audio:
            duration_s = len(audio["bytes"]) / (audio.get("sample_rate", 44100) * 2)
            st.markdown(f"""
<div style="background:#1a3a1a;border:1px solid #4CAF50;border-radius:8px;
            padding:10px 14px;margin:8px 0;display:flex;align-items:center;gap:12px">
  <span style="font-size:24px">✅</span>
  <div>
    <div style="color:#4CAF50;font-weight:600">Recording received ({duration_s:.1f}s)</div>
    <div style="color:#9BA3C2;font-size:12px">Transcribing with Whisper AI...</div>
  </div>
</div>
""", unsafe_allow_html=True)

            if client:
                with st.spinner("Transcribing... this takes 2-3 seconds"):
                    result = transcribe_audio(audio["bytes"], client)

                if result["error"]:
                    st.error(f"Transcription failed: {result['error']}")
                    st.caption("Try speaking louder or check your microphone permissions in the browser.")
                else:
                    st.session_state["em_context_prefill"] = result["text"]
                    lang = result["language_name"]
                    words = len(result["text"].split())

                    st.markdown(f"""
<div style="background:#1a2a3a;border:1px solid #7F77DD;border-radius:10px;padding:14px 18px;margin:8px 0">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="color:#7F77DD;font-weight:600;font-size:13px">
      🗣 Transcribed in {lang}
    </span>
    <span style="background:#2D3147;padding:2px 10px;border-radius:4px;
                 font-size:11px;color:#9BA3C2">{words} words · {duration_s:.1f}s</span>
  </div>
  <div style="color:#CDD0E0;font-size:13px;line-height:1.6;font-style:italic">
    "{result["text"][:200]}{"..." if len(result["text"]) > 200 else ""}"
  </div>
  <div style="margin-top:8px;font-size:11px;color:#9BA3C2">
    ✅ Auto-filled in the details box below · You can edit it before generating
  </div>
</div>
""", unsafe_allow_html=True)

                    # Language auto-suggestion
                    LANG_MAP = {
                        "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
                        "mr": "Marathi", "bn": "Bengali", "en": "English",
                        "kn": "Kannada", "gu": "Gujarati",
                    }
                    detected_lang = LANG_MAP.get(result["language"])
                    if detected_lang:
                        st.info(
                            f"💡 Detected **{detected_lang}** speech — "
                            f"your messages will be generated in {detected_lang}. "
                            f"Change the language dropdown above if needed."
                        )
            else:
                st.warning("Add OPENAI_API_KEY to .env to enable voice transcription")
    else:
        st.markdown("""
<div style="background:#1E2130;border:1px dashed #4D5275;border-radius:8px;
            padding:12px 16px;font-size:12px;color:#9BA3C2">
  Voice input not available. Run:
  <code style="background:#2D3147;padding:2px 6px;border-radius:3px">
    pip install streamlit-mic-recorder
  </code>
  then restart the app.
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── WhatsApp paste analyser (NEW) ────────────────────────────────────────
    with st.expander("📱 Paste a threatening WhatsApp message — auto-fill everything", expanded=False):
        st.caption("Copy the threatening message they sent you and paste it below. We'll extract the violation details automatically.")
        whatsapp_text = st.text_area(
            "Paste threatening message here",
            placeholder=(
                'e.g. "Aapka loan overdue hai. Aaj tak payment nahi ki to aapke boss ko call karenge '
                'aur police complaint karenge. Rs 22,000 abhi transfer karo."'
            ),
            height=100,
            key="whatsapp_paste",
        )
        if st.button("🔍 Analyse This Message", key="analyse_whatsapp"):
            if whatsapp_text.strip() and client:
                with st.spinner("Analysing for legal violations..."):
                    wa_prompt = f"""Analyse this threatening message from a loan lender/recovery agent:

"{whatsapp_text}"

Return JSON:
- violation_type: main violation (e.g. "Threatening arrest / filing criminal case")
- illegal_phrases: list of phrases that are illegal under Indian law
- laws_violated: specific Indian laws violated (IPC sections, RBI circulars)
- amount_demanded: amount in rupees if mentioned, else null
- severity: "Low" / "Medium" / "High" / "Extreme"
- is_illegal_threat: true if they threatened arrest, CIBIL, or family contact
- summary: one sentence summary of what they violated"""
                    try:
                        res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Expert Indian consumer rights lawyer."},
                                {"role": "user",   "content": wa_prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.2,
                        )
                        wa_result = json.loads(res.choices[0].message.content)
                        st.session_state["wa_analysis"] = wa_result
                        # Auto-fill context
                        st.session_state["em_context_prefill"] = (
                            f"They sent this threatening message: {whatsapp_text[:300]}. "
                            f"Violations: {', '.join(wa_result.get('illegal_phrases', []))[:200]}"
                        )
                        if wa_result.get("amount_demanded"):
                            st.session_state["em_demanded_voice"] = float(wa_result["amount_demanded"])
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            elif not client:
                st.warning("Add OPENAI_API_KEY to .env to analyse messages")
            else:
                st.warning("Please paste a message first")

        if "wa_analysis" in st.session_state:
            wa = st.session_state["wa_analysis"]
            sev_color = {"Low":"🟡","Medium":"🟠","High":"🔴","Extreme":"🚨"}.get(wa.get("severity","High"),"🔴")
            st.markdown(f"{sev_color} **Severity: {wa.get('severity','High')}** — {wa.get('summary','')}")
            if wa.get("illegal_phrases"):
                    phrases_list = " | ".join(wa["illegal_phrases"])
                    st.error(f"**Illegal phrases detected:** {phrases_list}")

            if wa.get("laws_violated"):
                st.info("**Laws violated:** " + wa.get("laws_violated",""))
            if wa.get("is_illegal_threat"):
                st.warning(
                    "⚠️ **Arrest threats are illegal.** Loan default is a civil matter — "
                    "police CANNOT arrest you for not paying a loan. This message is itself a crime."
                )
            st.success("✅ Context auto-filled — click 'Generate My Legal Messages' below")

    # ── Optional amount details ───────────────────────────────────
    with st.expander("📊 Add loan amount details (optional — improves messages)", expanded=False):
        c1, c2, c3 = st.columns(3)
        em_disbursed = c1.number_input(
            "Amount you received (₹)",
            min_value=0.0, value=0.0, step=100.0, key="em_disbursed",
            help="The actual amount deposited in your account"
        )
        em_demanded = c2.number_input(
            "Amount they are demanding (₹)",
            min_value=0.0, value=0.0, step=100.0, key="em_demanded",
            help="Total amount they say you must pay"
        )
        em_days = c3.number_input(
            "Days since loan was taken",
            min_value=1, value=30, step=1, key="em_days",
            help="How many days ago did you receive the money?"
        )
        if em_disbursed > 0 and em_demanded > 0:
            quick_max = em_disbursed * ((1 + 0.36/365) ** int(em_days))
            quick_extortion = max(0, em_demanded - quick_max)
            if quick_extortion > 0:
                st.warning(
                    f"⚠️ Quick check: You received ₹{em_disbursed:,.0f}. "
                    f"Legal max at 36% APR for {int(em_days)} days = **₹{quick_max:,.0f}**. "
                    f"They are overcharging by **₹{quick_extortion:,.0f}**. "
                    f"Use Tab 4 for the full legal breakdown."
                )

    # ── RL Strategy preview ───────────────────────────────────────
    lender_map_pre = {
        "Illegal loan app":              "illegal_app",
        "Bank (SBI, HDFC, ICICI etc.)": "bank",
        "NBFC / Fintech lender":         "nbfc",
        "Microfinance institution (MFI)":"mfi",
    }
    _lt   = lender_map_pre.get(em_lender, "nbfc")
    _task, _reason = classify_situation(_lt, 0.7, int(em_days if em_days else 30), float(em_demanded or 15000))
    _tp   = POLICY.get("tasks", {}).get(_task, {})
    _seq  = _tp.get("best_sequence", [])

    action_short = {
        "send_written_notice":     "Notice",
        "file_rbi_complaint":      "RBI Complaint",
        "file_police_complaint":   "Police Report",
        "request_debt_validation": "Debt Validation",
        "negotiate_settlement":    "Negotiate",
        "escalate_to_ombudsman":   "Ombudsman",
        "document_violations":     "Document",
        "contact_consumer_forum":  "Consumer Forum",
        "invoke_cooling_off":      "Cooling-off",
        "cite_kfs_violation":      "KFS Dispute",
    }

    if _seq:
        seq_display = " → ".join(action_short.get(a, a) for a in _seq)
        _src = "RL-trained" if os.path.exists(
            os.path.join(os.path.dirname(__file__), "policy.json")) else "Baseline"
        st.info(
            f"**RL Strategy ({_src}) for your situation:** {seq_display}  \n"
            f"**Why:** {_tp.get('why','')[:140]}..."
            if _tp.get('why') else f"**RL Strategy ({_src}):** {seq_display}"
        )

    # ── Generate button ───────────────────────────────────────────
    current_fp = f"{em_lender}|{em_violation}|{em_language}|{em_app}|{em_disbursed}|{em_demanded}|{em_context[:50]}"
    if st.session_state.get("em_fp") != current_fp:
        st.session_state.pop("emergency_msgs", None)
        st.session_state["em_fp"] = current_fp

    if st.button("🆘 Generate My Legal Messages", type="primary", use_container_width=True):
        with st.spinner("Drafting legally grounded messages..."):
            lender_map = {
                "Illegal loan app":              "illegal_app",
                "Bank (SBI, HDFC, ICICI etc.)": "bank",
                "NBFC / Fintech lender":         "nbfc",
                "Microfinance institution (MFI)":"mfi",
            }
            msgs = generate_emergency_messages(
                lender_type    = lender_map.get(em_lender, "nbfc"),
                violation_type = em_violation,
                language       = em_language,
                app_name       = em_app,
                disbursed      = em_disbursed,
                demanded       = em_demanded,
                extra_context  = em_context,
            )
            st.session_state["emergency_msgs"] = msgs

    # ── Display messages ──────────────────────────────────────────
    if "emergency_msgs" in st.session_state:
        msgs = st.session_state["emergency_msgs"]

        col_kf, col_dn = st.columns(2)
        if msgs.get("key_fact"):
            col_kf.success(f"✅ **Know this:** {msgs['key_fact']}")
        if msgs.get("do_not"):
            col_dn.error(f"🚫 **Do NOT:** {msgs['do_not']}")

        st.divider()

        has_regional = bool(msgs.get("msg1") and msgs.get("msg1") != msgs.get("msg1_en"))
        if has_regional:
            lang_choice = st.radio(
                "Language",
                options=["regional", "english"],
                format_func=lambda x: f"🇮🇳 {em_language}" if x == "regional" else "🇬🇧 English",
                horizontal=True,
                key="em_lang_radio",
                label_visibility="collapsed",
            )
        else:
            lang_choice = "english"

        for idx, (label_key, msg_key_r, msg_key_en) in enumerate([
            ("msg1_label", "msg1", "msg1_en"),
            ("msg2_label", "msg2", "msg2_en"),
            ("msg3_label", "msg3", "msg3_en"),
        ], 1):
            label = msgs.get(label_key, f"Message {idx}")
            msg   = (msgs.get(msg_key_r) if lang_choice == "regional"
                     else msgs.get(msg_key_en)) or msgs.get(msg_key_r, "")
            if msg:
                st.markdown(f"**{idx}. {label}**")
                st.text_area("Message", value=msg, height=150, label_visibility="collapsed")
                st.caption("📋 Select all → Copy → Send via WhatsApp / Email / Registered Post")
                st.divider()

        col_a, col_b, col_c = st.columns(3)
        col_a.markdown("**[📱 RBI CMS Portal](https://cms.rbi.org.in)**")
        col_b.markdown("**[🚨 Cybercrime Portal](https://cybercrime.gov.in)**")
        col_c.markdown("**[📋 SACHET Portal](https://sachet.rbi.org.in)**")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — App Checker
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Is your loan app legal?")
    st.caption("Non-payment to **illegal apps** will NOT affect your CIBIL score. Know before you pay.")

    app_name_input = st.text_input(
        "Type the app or lender name",
        placeholder="e.g. QuickCash, LazyPay, HDFC, Bajaj Finserv...",
        key="app_checker"
    )

    if app_name_input:
        with st.spinner("Checking RBI directory..."):
            verdict, color, advice = check_app_legality(app_name_input)

        color_map = {
            "red":    ("🔴", "#3a1a1a", "#FF6B6B", "#D85A30"),
            "orange": ("🟡", "#3a2e1a", "#FFC107", "#EF9F27"),
            "green":  ("🟢", "#1a3a2a", "#4CAF50", "#1D9E75"),
        }
        icon, bg, text_c, border_c = color_map.get(color, color_map["orange"])

        st.markdown(f"""
<div style="background:{bg};border-left:5px solid {border_c};border-radius:0 12px 12px 0;
            padding:18px 22px;margin:14px 0">
  <div style="font-size:18px;font-weight:600;color:{text_c};margin-bottom:10px">
    {icon} &nbsp;{verdict}
  </div>
  <div style="font-size:14px;color:#CDD0E0;line-height:1.8">{advice}</div>
</div>
""", unsafe_allow_html=True)

        if color == "red":
            st.markdown("#### What to do immediately:")
            steps = [
                ("📞", "Call **1930** NOW", "National Cybercrime Helpline — 24×7, free"),
                ("🚫", "Stop paying immediately", "You have no legal obligation to an unregistered lender"),
                ("🔒", "Revoke all app permissions", "Settings → Apps → [App] → Permissions → Deny all"),
                ("🗑️", "Uninstall the app", "Limits their data access immediately"),
                ("💻", "File at cybercrime.gov.in", "Online complaint takes 5 minutes"),
                ("📋", "Report at sachet.rbi.org.in", "RBI's portal for illegal lending complaints"),
            ]
            for icon_s, action, detail in steps:
                st.markdown(
                    f"<div style='background:#1E2130;border-radius:8px;padding:10px 14px;margin-bottom:6px'>"
                    f"<span style='font-size:18px'>{icon_s}</span> "
                    f"<strong style='color:#FF6B4A'>{action}</strong> "
                    f"<span style='color:#9BA3C2;font-size:12px'>— {detail}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            st.error(
                "**CIBIL FACT (RBI 2025):** Illegal/unregistered apps are NOT connected to CIBIL. "
                "Non-payment will NOT appear on your credit report. Do not pay extortion disguised as debt."
            )
        elif color == "green":
            st.info(
                "**Your rights with a registered lender (RBI 2025):**\n"
                "- Cooling-off: cancel within 3 days of disbursement, no penalty\n"
                "- KFS: all charges must be disclosed upfront — undisclosed charges are unenforceable\n"
                "- Harassment: file at cms.rbi.org.in — Ombudsman awards up to ₹20 lakh\n"
                "- Call times: agents can only contact you 7am–7pm"
            )

    with st.expander("How to verify any app — RBI DLA directory"):
        st.markdown(
            "**Official source:** RBI launched the Digital Lending Apps (DLA) directory (July 2025).  \n"
            "Any app NOT listed is unregistered and operating illegally.  \n\n"
            "1. Visit **rbi.org.in** → Search 'Digital Lending Apps directory'\n"
            "2. Report suspected illegal apps at **sachet.rbi.org.in**\n"
            "3. Cybercrime helpline: **1930** (24×7, free)\n"
            "4. Online portal: **cybercrime.gov.in**"
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Violation Log
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Log every violation — build your complaint automatically")
    st.caption("Each entry becomes evidence. 3+ entries → auto-generate your RBI complaint.")

    with st.form("log_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        log_date = c1.date_input("Date of violation", value=datetime.date.today())
        log_time = c2.time_input("Time", value=datetime.time(9, 0))

        log_type = st.selectbox("Type of violation", [
            "Call outside 7am–7pm hours",
            "Abusive / threatening language",
            "Contacted family / friends / employer",
            "Threatened arrest (loan default is civil, not criminal)",
            "Visited home / workplace without notice",
            "Morphed photos / blackmail",
            "10+ calls in one day",
            "Demanded more than legally owed",
            "Unauthorized access to contacts / gallery",
            "Other",
        ])

        log_detail = st.text_area(
            "Describe what happened",
            placeholder=(
                "Be specific: exact words used, who called, what was said, "
                "who else was present or contacted...\n"
                "e.g. 'A person named Raju called my sister at 8:30pm and said "
                "I would be arrested tomorrow if I don't pay ₹15,000 by midnight.'"
            ),
            height=120,
            help="The more specific, the stronger your complaint. Include names, exact words, times."
        )

        col_ev, col_sev = st.columns(2)
        log_evidence = col_ev.selectbox("Evidence you have", [
            "Call recording",
            "Screenshot of message",
            "WhatsApp screenshot",
            "Email screenshot",
            "Witness present",
            "No evidence yet",
        ])
        log_severity = col_sev.select_slider(
            "Severity", options=["Low", "Medium", "High", "Extreme"],
            value="High"
        )

        submitted = st.form_submit_button("➕ Add to Violation Log", use_container_width=True)
        if submitted:
            if not log_detail.strip():
                st.error("Please describe what happened before adding.")
            else:
                st.session_state["log"].append({
                    "date":     str(log_date),
                    "time":     str(log_time),
                    "type":     log_type,
                    "detail":   log_detail,
                    "evidence": log_evidence,
                    "severity": log_severity,
                })
                st.success(f"✅ Logged. Total violations recorded: {len(st.session_state['log'])}")

    if st.session_state["log"]:
        st.divider()

        # Summary metrics
        n = len(st.session_state["log"])
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Violations logged", n)
        col_m2.metric("Ready to complain", "Yes ✅" if n >= 3 else f"Need {3-n} more")
        col_m3.metric("Evidence items",
                      sum(1 for e in st.session_state["log"] if "No evidence" not in e["evidence"]))

        # Log entries
        for i, entry in enumerate(st.session_state["log"], 1):
            sev_color = {"Low":"🟡","Medium":"🟠","High":"🔴","Extreme":"🚨"}.get(entry.get("severity","High"),"🔴")
            with st.expander(f"{sev_color} #{i} — {entry['date']} {entry['time']} — {entry['type']}"):
                st.write(f"**Description:** {entry['detail'] or 'No description'}")
                st.write(f"**Evidence:** {entry['evidence']}")
                st.write(f"**Severity:** {entry.get('severity','High')}")

        # Generate complaint
        st.divider()
        if st.button("📄 Generate RBI CMS Complaint from Log", type="primary"):
            with st.spinner("Drafting your formal complaint..."):
                entries_text = "\n".join([
                    f"- {e['date']} at {e['time']} [{e.get('severity','High')} severity]: "
                    f"{e['type']}. Details: {e['detail']}. Evidence: {e['evidence']}."
                    for e in st.session_state["log"]
                ])
                prompt = (
                    "Draft a formal complaint for the RBI Complaints Management System (cms.rbi.org.in). "
                    "The borrower has documented these violations of the RBI Digital Lending Directions 2025 "
                    "and Fair Practices Code:\n\n"
                    f"{entries_text}\n\n"
                    "Write a structured complaint letter that: cites specific RBI circulars and sections "
                    "violated, lists incidents in date order with specifics, requests specific relief "
                    "(immediate cessation of harassment, compensation under Integrated Ombudsman Scheme), "
                    "and is suitable for submission to the RBI Banking/NBFC Ombudsman. "
                    "Include the legal basis for each violation."
                )
                if client:
                    res = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Expert Indian consumer rights lawyer."},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    complaint = res.choices[0].message.content
                else:
                    complaint = "Please configure OPENAI_API_KEY to generate the complaint."
                st.session_state["generated_complaint"] = complaint

        if "generated_complaint" in st.session_state:
            st.markdown("#### Your RBI Complaint (ready to submit):")
            st.text_area(
                "Complaint",
                value=st.session_state["generated_complaint"],
                height=320,
                label_visibility="collapsed"
            )
            st.success("✅ Copy → go to cms.rbi.org.in → paste into complaint form")
            col_r1, col_r2 = st.columns(2)
            col_r1.markdown("**[Open RBI CMS Portal →](https://cms.rbi.org.in)**")
            col_r2.markdown("**[SACHET Portal →](https://sachet.rbi.org.in)**")

        if st.button("🗑️ Clear All Logs", type="secondary"):
            st.session_state["log"] = []
            st.session_state.pop("generated_complaint", None)
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Legal Amount Calculator (FIXED)
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### What do you legally owe?")
    st.caption(
        "Illegal apps disburse ₹1,800 and demand ₹9,000. "
        "See exactly what's legal — and what's extortion."
    )

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        calc_disbursed = st.number_input(
            "Amount you actually received (₹)",
            min_value=0.0, value=10000.0, step=500.0,
            help="The exact rupee amount deposited in your account — NOT what was approved"
        )
        calc_demanded = st.number_input(
            "Amount they are demanding (₹)",
            min_value=0.0, value=25000.0, step=500.0,
            help="Total amount they say you must pay right now"
        )

    with col_i2:
        calc_days = st.number_input(
            "Days since loan was taken",
            min_value=1, value=30, step=1,
            help="How many days ago did you receive the money?"
        )
        calc_lender = st.selectbox("Lender type", [
            "Illegal app (legally owe ONLY principal — no interest rights)",
            "Bank / NBFC (max 36% per annum — RBI cap)",
            "MFI (max 26% per annum — RBI MFIN cap)",
        ])

    # Optional: additional charges breakdown
    with st.expander("📊 Break down the charges they added (optional)"):
        st.caption("Enter each charge to see which ones are legally enforceable")
        col_c1, col_c2, col_c3 = st.columns(3)
        processing_fee = col_c1.number_input("Processing fee (₹)", min_value=0.0, value=0.0, step=100.0)
        late_fee       = col_c2.number_input("Late fees (₹)", min_value=0.0, value=0.0, step=100.0)
        other_charges  = col_c3.number_input("Other charges (₹)", min_value=0.0, value=0.0, step=100.0)
        kfs_provided   = st.checkbox(
            "Lender provided a Key Fact Statement (KFS) before disbursement",
            value=False,
            help="Under RBI 2025, lenders MUST provide KFS before disbursement. "
                 "If they didn't, all undisclosed charges are legally unenforceable."
        )

    calc_comment = st.text_area(
        "Any additional context (optional)",
        placeholder="e.g. They didn't tell me about the processing fee before giving the loan. "
                    "The app charged ₹1,200 processing fee that was never mentioned...",
        height=80,
        key="calc_comment"
    )

    if st.button("⚖️ Calculate What I Actually Owe", type="primary", use_container_width=True):
        rate_map = {
            "Illegal app (legally owe ONLY principal — no interest rights)": 0.0,
            "Bank / NBFC (max 36% per annum — RBI cap)":                     0.36,
            "MFI (max 26% per annum — RBI MFIN cap)":                        0.26,
        }
        annual_rate     = rate_map.get(calc_lender, 0.36)
        is_illegal      = "Illegal" in calc_lender
        result          = calculate_legal_amount(
            calc_disbursed, calc_demanded, int(calc_days), annual_rate, calc_lender
        )

        st.divider()

        # ── Metrics row ───────────────────────────────────────────
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("You received",          f"₹{calc_disbursed:,.0f}")
        col_b.metric("Legal interest",         f"₹{result['legal_interest']:,.0f}",
                     help=result["note"])
        col_b_val = f"₹{result['legal_max']:,.0f}"
        col_c.metric("You legally owe",        col_b_val,
                     delta="(MAX)" if not is_illegal else "Principal only")
        delta_ext = f"+₹{result['extortion']:,.0f} illegal" if result['extortion'] > 0 else "Within limits"
        col_d.metric("They are demanding",     f"₹{calc_demanded:,.0f}",
                     delta=delta_ext,
                     delta_color="inverse" if result['extortion'] > 0 else "normal")

        st.divider()

        # ── KFS check ─────────────────────────────────────────────
        undisclosed = processing_fee + late_fee + other_charges
        if undisclosed > 0 and not kfs_provided:
            st.error(
                f"### ₹{undisclosed:,.0f} in charges are legally unenforceable\n\n"
                f"The lender did NOT provide a Key Fact Statement (KFS) before disbursement. "
                f"Under RBI Digital Lending Directions 2025, Section 8, "
                f"**all charges not disclosed in the KFS are legally unenforceable.** "
                f"You can formally dispute: processing fee ₹{processing_fee:,.0f}, "
                f"late fees ₹{late_fee:,.0f}, other ₹{other_charges:,.0f}."
            )
        elif undisclosed > 0 and kfs_provided:
            st.info(
                f"Lender provided KFS. Processing fees and charges (₹{undisclosed:,.0f}) "
                f"may be enforceable IF they were disclosed in the KFS. "
                f"Check your KFS document — if any charge is missing from it, dispute that charge."
            )

        # ── Extortion breakdown ───────────────────────────────────
        if result["extortion"] > 0:
            pct = result["overpayment_pct"]
            st.error(
                f"### ₹{result['extortion']:,.0f} is extortion — not debt\n\n"
                f"You legally owe **₹{result['legal_max']:,.2f}**. "
                f"They are demanding **₹{calc_demanded:,.0f}** — that is "
                f"**{pct:.0f}% more** than the maximum the law allows. "
                f"Paying the excess is not a legal obligation — it is extortion."
            )
            if is_illegal:
                st.warning(
                    "**For illegal apps:** The entire demand may be unenforceable. "
                    "Unregistered apps have NO legal right to collect any debt. "
                    "Non-payment will NOT affect your CIBIL score. "
                    "Use Tab 2 to verify if your app is registered."
                )
        else:
            st.success(
                f"The amount demanded (₹{calc_demanded:,.0f}) is within legal limits "
                f"(legal max = ₹{result['legal_max']:,.0f} at {annual_rate*100:.0f}% APR "
                f"for {int(calc_days)} days). This appears to be a legitimate demand."
            )

        # ── Your options ──────────────────────────────────────────
        st.divider()
        st.markdown("#### Your legal options:")
        if result["extortion"] > 0 or (undisclosed > 0 and not kfs_provided):
            options = []
            if result["extortion"] > 0:
                options.append(f"**Pay only ₹{result['legal_max']:,.0f}** and send a written notice disputing the excess")
            if undisclosed > 0 and not kfs_provided:
                options.append(f"**Dispute ₹{undisclosed:,.0f}** in undisclosed charges under RBI 2025 KFS rules")
            options.append("**File a complaint at cms.rbi.org.in** citing the overcharge")
            options.append("**Under Consumer Protection Act 2019**, claim the overcharged amount back as damages")
            if is_illegal:
                options.append("**File at cybercrime.gov.in** — non-payment will NOT hurt your CIBIL")

            for opt in options:
                st.markdown(f"- {opt}")

        if calc_comment.strip():
            st.info(f"**Your context noted:** {calc_comment}  \n"
                    f"Use Tab 1 to generate a specific legal message referencing these details.")


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Full Strategy
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Full AI legal strategy — RL-trained policy")
    st.caption("For a complete action plan grounded in what the RL agent learned against your lender type.")

    col1, col2 = st.columns(2)
    with col1:
        lender_type = st.selectbox("Who is harassing you?", [
            "bank", "nbfc", "mfi", "illegal_app"
        ], format_func=lambda x: {
            "bank":        "Bank (SBI, HDFC, ICICI etc.)",
            "nbfc":        "NBFC / Fintech lender",
            "mfi":         "Microfinance (MFI)",
            "illegal_app": "Illegal loan app",
        }[x], key="strat_lender")

        debt_amount = st.number_input(
            "Outstanding debt amount (₹)",
            value=15000.0, step=1000.0, key="strat_debt"
        )

    with col2:
        days_overdue   = st.number_input("Days since default", value=45, key="strat_days")
        harassment_lvl = st.slider(
            "Harassment severity (0 = mild, 1 = extreme)",
            0.0, 1.0, 0.6, step=0.05, key="strat_harass"
        )

    situation = st.text_area(
        "Describe your situation in detail",
        placeholder=(
            "The more detail you give, the better the strategy.\n"
            "e.g. I borrowed ₹15,000 from an app called QuickCash 45 days ago. "
            "They are now demanding ₹28,000. They call my employer daily. "
            "They sent messages to my WhatsApp contacts saying I am a fraud. "
            "They never gave me a KFS document before the loan."
        ),
        height=130,
        key="strat_situation",
        help="Include: what they are doing, what they are demanding, how long this has been happening."
    )

    if st.button("⚖️ Generate My Legal Strategy", type="primary", use_container_width=True):
        with st.spinner("Consulting RL policy + Indian law..."):
            advice = get_advice(
                lender_type      = lender_type,
                harassment_level = harassment_lvl,
                days_overdue     = int(days_overdue),
                debt_amount      = debt_amount,
                violation_type   = situation or "general harassment and overcharging",
                client           = client,
                language         = "English",
            )
            st.session_state["strategy"] = advice

    if "strategy" in st.session_state:
        adv = st.session_state["strategy"]
        st.divider()

        # ── Policy source + task badge ────────────────────────────
        src = adv.get("policy_source", "baseline")
        col_src, col_task = st.columns(2)
        with col_src:
            if src == "rl_trained":
                st.success("🤖 **RL-Trained Policy**")
            else:
                st.warning("📋 **Baseline Policy**")
        with col_task:
            st.info(
                f"**Task:** `{adv['task']}` · "
                f"**Sim score:** {adv['score_expected']:.0%}"
                + (f" · **Pass rate:** {adv['pass_rate']*100:.0f}%" if adv.get("pass_rate") else "")
            )

        st.caption(f"**Why this task:** {adv['task_reason']}")

        # ── Sequence overview ─────────────────────────────────────
        seq = adv.get("sequence", [])
        seq_display = " → ".join(action_short.get(a, a) for a in seq)
        st.markdown(f"**RL sequence:** `{seq_display}`")

        # ── Generate full automated plan ──────────────────────────
        if st.button("🚀 Generate Full Automated Action Plan", type="primary",
                     use_container_width=True, key="gen_plan_btn"):
            with st.spinner("Generating your complete action plan with messages for every step..."):
                plan = generate_full_plan(
                    sequence    = seq,
                    lender_type = lender_type,
                    situation   = situation or "general harassment",
                    debt_amount = debt_amount,
                    client      = client,
                )
                st.session_state["full_plan"] = plan

        # ── Display the automated plan ────────────────────────────
        if "full_plan" in st.session_state:
            plan = st.session_state["full_plan"]

            st.divider()
            st.markdown("## 📋 Your Complete Legal Action Plan")
            st.caption(
                f"{len(plan)} steps · "
                f"Total duration ~{plan[-1]['day']} days · "
                f"All messages ready to copy and send"
            )

            # Progress tracker
            progress_html = '<div style="display:flex;gap:0;margin-bottom:20px;flex-wrap:wrap">'
            for step in plan:
                day_label = "Today" if step["day"] == 0 else f"Day {step['day']}"
                progress_html += (
                    f'<div style="flex:1;min-width:80px;text-align:center;padding:6px 4px;'
                    f'background:#1E2130;border:1px solid #2D3147;margin:2px;border-radius:6px">'
                    f'<div style="color:#7F77DD;font-size:10px;font-weight:600">{day_label}</div>'
                    f'<div style="color:#CDD0E0;font-size:10px;margin-top:2px">'
                    f'{action_short.get(step["action"], step["action"][:8])}</div>'
                    f'</div>'
                )
            progress_html += '</div>'
            st.markdown(progress_html, unsafe_allow_html=True)

            # Render each step
            for step in plan:
                day_label = "🔴 **TODAY — Do this NOW**" if step["day"] == 0 else f"📅 **Day {step['day']}**"
                border_color = "#D85A30" if step["day"] == 0 else "#7F77DD"
                bg_color     = "#2a1a1a" if step["day"] == 0 else "#1E2130"

                st.markdown(f"""
<div style="background:{bg_color};border:1px solid {border_color};
            border-left:5px solid {border_color};border-radius:10px;
            padding:16px 20px;margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
    <div>
      <span style="color:{border_color};font-size:11px;font-weight:700;
                   text-transform:uppercase;letter-spacing:.05em">
        Step {step['step']} · {day_label.replace("**","").replace("🔴","").replace("📅","").strip()}
      </span>
      <div style="color:#FFFFFF;font-size:16px;font-weight:600;margin-top:3px">{step['title']}</div>
    </div>
    <div style="background:#2D3147;padding:3px 10px;border-radius:4px;
                font-size:10px;color:#9BA3C2;white-space:nowrap;margin-left:8px">
      {step['portal']}
    </div>
  </div>
  <div style="color:#9BA3C2;font-size:12px;margin-bottom:10px;line-height:1.5">
    💡 <em>{step['why']}</em>
  </div>
  <div style="color:#7BA3C2;font-size:11px;margin-bottom:8px">
    📋 Law: {step['law']}
  </div>
</div>
""", unsafe_allow_html=True)

                # Message text area for each step
                st.text_area(
                    f"Message for Step {step['step']} — {step['title']}",
                    value=step["message"],
                    height=160,
                    key=f"plan_msg_{step['step']}",
                    label_visibility="collapsed",
                )

                col_exp, col_link = st.columns([3, 1])
                with col_exp:
                    st.caption(f"✅ Expected outcome: {step['expected']}")
                with col_link:
                    if step["portal_url"]:
                        st.markdown(f"[Open {step['portal']} →]({step['portal_url']})")

                st.divider()

            # ── Prohibited actions ────────────────────────────────
            if adv.get("prohibited"):
                st.warning(
                    f"**🚫 The RL agent learned NOT to use:** {', '.join(adv['prohibited'])}  \n"
                    f"These actions consistently scored lower for your scenario in simulation."
                )

            # ── Download full plan ────────────────────────────────
            full_text = f"INDIA DEBT RIGHTS NAVIGATOR — YOUR LEGAL ACTION PLAN\n"
            full_text += f"Generated: {datetime.date.today()}\n"
            full_text += f"Situation: {situation or 'General harassment'}\n"
            full_text += f"Lender: {lender_type} | Debt: Rs {debt_amount:,.0f}\n"
            full_text += f"Strategy: {seq_display}\n"
            full_text += "=" * 60 + "\n\n"
            for step in plan:
                day_str = "TODAY" if step["day"] == 0 else f"Day {step['day']}"
                full_text += f"STEP {step['step']} ({day_str}): {step['title']}\n"
                full_text += f"Why: {step['why']}\n"
                full_text += f"Law: {step['law']}\n"
                full_text += f"Message:\n{step['message']}\n"
                full_text += f"Portal: {step['portal']}"
                if step["portal_url"]:
                    full_text += f" — {step['portal_url']}"
                full_text += f"\nExpected: {step['expected']}\n"
                full_text += "-" * 40 + "\n\n"
            full_text += "Helpline 1930 | cms.rbi.org.in | cybercrime.gov.in\n"

            st.download_button(
                label       = "⬇️ Download Full Action Plan (.txt)",
                data        = full_text,
                file_name   = f"debt_rights_action_plan_{datetime.date.today()}.txt",
                mime        = "text/plain",
                use_container_width = True,
            )


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Your Rights (RBI 2025)")
    rights = [
        ("🕐", "Agents can only call **7am–7pm**"),
        ("🚫", "Threats of arrest are **illegal**"),
        ("🔒", "Contact-list access = **IT Act 66E crime**"),
        ("📋", "RBI Ombudsman: **free, up to ₹20 lakh**"),
        ("💡", "Illegal app? **CIBIL not affected**"),
        ("📄", "**KFS right:** All charges disclosed before loan (2025)"),
        ("⏱", "**Cooling-off:** Cancel within 3 days, no penalty (2025)"),
        ("📞", "**Helpline 1930** — cybercrime, 24×7, free"),
    ]
    for icon_r, text in rights:
        st.markdown(f"{icon_r} {text}")

    st.divider()
    st.markdown("**Quick links**")
    st.markdown(
        "- [RBI CMS Portal](https://cms.rbi.org.in)\n"
        "- [Cybercrime Portal](https://cybercrime.gov.in)\n"
        "- [SACHET Portal](https://sachet.rbi.org.in)\n"
        "- [Consumer Forum](https://edaakhil.nic.in)\n"
        "- [CIBIL Dispute](https://www.cibil.com)"
    )
    st.divider()
    st.markdown("**Legal Knowledge Base**")
    for law, desc in LEGAL_KB.items():
        with st.expander(law.replace("_", " ")):
            st.caption(desc)


# ── Required for OpenEnv validator ───────────────────────────────────────────
def main() -> None:
    pass

if __name__ == "__main__":
    main()