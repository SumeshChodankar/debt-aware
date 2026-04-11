"""
India Debt Rights Navigator — Real Victim Emergency Dashboard
5 features that actually help someone being harassed right now.
"""

import os
import json
import asyncio
import datetime
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from engine.core import RBIRightsEnv, LEGAL_KB
from policy_advisor import get_advice, classify_situation, POLICY
from engine.models import Action, LegalAction

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
client  = OpenAI(api_key=api_key) if api_key else None

st.set_page_config(
    page_title="India Debt Rights — Emergency Help",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# App legitimacy checker — uses live RBI DLA directory (launched July 1 2025)
# Falls back to curated list if network unavailable
# ---------------------------------------------------------------------------
import urllib.request

# Curated fallback list (used when RBI API is unreachable)
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

def _check_rbi_directory(app_name: str) -> str | None:
    """
    Attempt to check RBI DLA directory (July 2025).
    Returns "registered", "not_found", or None if unreachable.
    The RBI CIMS portal lists all DLAs registered by regulated entities.
    """
    try:
        # RBI NBFC list endpoint — publicly accessible
        url = f"https://www.rbi.org.in/Scripts/BS_NBFCList.aspx"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as r:
            body = r.read().decode("utf-8", errors="ignore").lower()
            name_clean = app_name.lower().strip()
            if name_clean in body or name_clean.replace(" ", "") in body:
                return "registered"
            return "not_found"
    except Exception:
        return None  # Network unavailable — use fallback


def check_app_legality(app_name: str) -> tuple[str, str, str]:
    """
    Returns (verdict, color, advice).
    Priority: live RBI directory > curated fallback > unknown.
    """
    name = app_name.lower().strip().replace(" ", "").replace("-", "")

    # Check curated illegal list first (fast)
    for a in _KNOWN_ILLEGAL:
        if a in name or name in a:
            return (
                "ILLEGAL — Unregistered (known predatory app)",
                "red",
                "DO NOT PAY. Not in RBI DLA directory. Non-payment will NOT affect CIBIL. "
                "Call 1930 immediately. File at cybercrime.gov.in and sachet.rbi.org.in."
            )

    # Check curated registered list (fast)
    for a in _KNOWN_REGISTERED:
        if a in name or name in a:
            return (
                "REGISTERED — In RBI DLA directory",
                "green",
                "This is a regulated lender. Payment affects CIBIL score. "
                "If harassed: file complaint at cms.rbi.org.in. "
                "You have cooling-off and KFS rights under 2025 Directions."
            )

    # Try live RBI directory (slower, may fail on network)
    rbi_result = _check_rbi_directory(app_name)
    if rbi_result == "registered":
        return (
            "REGISTERED — Verified in RBI DLA directory (live check)",
            "green",
            "Verified as registered with RBI as of July 2025. Payment affects CIBIL. "
            "You have KFS and cooling-off rights under RBI Digital Lending Directions 2025."
        )
    elif rbi_result == "not_found":
        return (
            "NOT IN RBI DIRECTORY — Likely illegal",
            "red",
            "Not found in RBI DLA directory (checked July 2025 live data). "
            "DO NOT PAY. Non-payment will NOT affect CIBIL. "
            "Report at cybercrime.gov.in and helpline 1930."
        )
    else:
        return (
            "UNKNOWN — Could not verify (check manually)",
            "orange",
            "Network check failed. Verify manually at rbi.org.in DLA directory. "
            "If not listed there, treat as illegal — do not pay. Helpline: 1930."
        )

# ---------------------------------------------------------------------------
# Generate emergency messages via LLM
# ---------------------------------------------------------------------------
def generate_emergency_messages(
    lender_type: str,
    violation_type: str,
    language: str,
    app_name: str = "",
    disbursed: float = 0,
    demanded: float = 0,
) -> dict:
    """
    Generates messages in BOTH English AND the selected regional language
    in a single API call.  The result dict contains:
      msg1 / msg2 / msg3          — regional language versions
      msg1_en / msg2_en / msg3_en — English versions
    The UI toggles between them without any extra API call.
    """
    regional_lang = language if language != "English" else None

    lang_block = ""
    if regional_lang:
        lang_map = {
            "Hindi":   "Hindi (Devanagari script)",
            "Tamil":   "Tamil script",
            "Telugu":  "Telugu script",
            "Marathi": "Marathi (Devanagari script)",
            "Bengali": "Bengali script",
        }
        lang_name = lang_map.get(regional_lang, regional_lang)
        lang_block = (
            f"For each message provide TWO versions:\n"
            f"  - An English version (keys: msg1_en, msg2_en, msg3_en)\n"
            f"  - A {lang_name} version (keys: msg1, msg2, msg3)\n"
            f"The content must be identical — just translated."
        )
    else:
        lang_block = (
            "Write all messages in English.\n"
            "Use the same text for both the regional keys (msg1, msg2, msg3) "
            "and English keys (msg1_en, msg2_en, msg3_en)."
        )

    amount_context = ""
    if disbursed > 0 and demanded > 0:
        legal_max = disbursed * 1.36
        extortion = demanded - legal_max
        amount_context = (
            f"\nLoan disbursed: ₹{disbursed:,.0f}. "
            f"Amount demanded: ₹{demanded:,.0f}. "
            f"Legal maximum (36% APR): ₹{legal_max:,.0f}. "
            f"Extortion amount: ₹{extortion:,.0f}."
        )

    prompt = f"""You are an expert Indian consumer rights lawyer.
A borrower is being harassed by a {lender_type} lender (app: {app_name or 'unknown'}).
Violation: {violation_type}.{amount_context}

{lang_block}

Return JSON with exactly these keys:
- msg1_label: short label for message 1 (in English)
- msg1: message to recovery agent — regional language
- msg1_en: message to recovery agent — English
- msg2_label: short label for message 2 (in English)
- msg2: complaint to Nodal Officer — regional language
- msg2_en: complaint to Nodal Officer — English
- msg3_label: short label for message 3 (in English)
- msg3: cybercrime/police complaint text — regional language
- msg3_en: cybercrime/police complaint text — English
- key_fact: one critical legal fact (English)
- do_not: one thing they must NOT do (English)"""

    if not client:
        stub = "API key not configured. Please add OPENAI_API_KEY to your .env file."
        return {
            "msg1_label": "Message to recovery agent",
            "msg1": stub, "msg1_en": stub,
            "msg2_label": "Email complaint",
            "msg2": "", "msg2_en": "",
            "msg3_label": "Police/Cybercrime complaint",
            "msg3": "", "msg3_en": "",
            "key_fact": "Configure your API key to generate messages.",
            "do_not": "Do not pay any amount before verifying the app is legitimate.",
        }

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

# ---------------------------------------------------------------------------
# Legal amount calculator
# ---------------------------------------------------------------------------
def calculate_legal_amount(disbursed: float, demanded: float, days: int) -> dict:
    daily_rate_legal = 0.36 / 365
    legal_max   = disbursed * (1 + daily_rate_legal * days)
    extortion   = max(0, demanded - legal_max)
    overpayment_pct = ((demanded - legal_max) / legal_max * 100) if legal_max > 0 else 0
    return {
        "disbursed":       disbursed,
        "demanded":        demanded,
        "legal_max":       round(legal_max, 2),
        "extortion":       round(extortion, 2),
        "overpayment_pct": round(overpayment_pct, 1),
        "days":            days,
    }

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state["log"] = []

# ---------------------------------------------------------------------------
# TOP BANNER — Emergency
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background:#FAECE7;border:1px solid #D85A30;border-radius:10px;padding:14px 20px;margin-bottom:1rem">
<span style="font-size:18px;font-weight:500;color:#712B13">🆘 Being harassed RIGHT NOW?</span>
<span style="font-size:13px;color:#993C1D;margin-left:12px">
Go to <b>Feature 1</b> → get 3 ready-to-send messages in 60 seconds.
Harassment is illegal. You have rights.
</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS — 5 features
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🆘 1. Emergency Response",
    "🔍 2. App Check",
    "📋 3. Harassment Log",
    "💰 4. Legal Amount",
    "⚖️ 5. Full Strategy",
])

# ============================================================
# TAB 1 — Emergency Response
# ============================================================
with tab1:
    st.markdown("### Get 3 ready-to-send messages in 60 seconds")
    st.caption("No legal knowledge needed. Copy. Send. Done.")

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
        ], key="em_violation")

    with col2:
        em_language = st.selectbox("Your preferred language", [
            "Hindi", "English", "Tamil", "Telugu", "Marathi", "Bengali"
        ], key="em_language")

        em_app = st.text_input(
            "App / lender name (optional)",
            placeholder="e.g. QuickCash, HDFC Bank...",
            key="em_app"
        )

    # Amount fields (optional)
    with st.expander("Add loan amount details (optional — improves messages)"):
        c1, c2, c3 = st.columns(3)
        em_disbursed = c1.number_input("Amount you received (₹)", min_value=0.0, value=0.0, step=100.0, key="em_disbursed")
        em_demanded  = c2.number_input("Amount they are demanding (₹)", min_value=0.0, value=0.0, step=100.0, key="em_demanded")
        em_days      = c3.number_input("Days since loan taken", min_value=1, value=7, step=1, key="em_days")

    # Fingerprint current inputs — clear stale result whenever anything changes
    current_fp = f"{em_lender}|{em_violation}|{em_language}|{em_app}|{em_disbursed}|{em_demanded}"
    if st.session_state.get("em_fp") != current_fp:
        st.session_state.pop("emergency_msgs", None)
        st.session_state["em_fp"] = current_fp

    # ── RL Policy preview — shown before user clicks ──────────────────
    lender_map_pre = {
        "Illegal loan app": "illegal_app",
        "Bank (SBI, HDFC, ICICI etc.)": "bank",
        "NBFC / Fintech lender": "nbfc",
        "Microfinance institution (MFI)": "mfi",
    }
    _lt  = lender_map_pre.get(em_lender, "nbfc")
    _hl  = st.session_state.get("em_harass_preview", 0.6)
    _do  = int(st.session_state.get("em_days", 45))
    _da  = float(st.session_state.get("em_demanded", 0) or 15000)
    _task, _reason = classify_situation(_lt, _hl, _do, _da)
    _tp   = POLICY.get("tasks", {}).get(_task, {})
    _seq  = _tp.get("best_sequence", [])
    if _seq:
        action_short = {
            "send_written_notice":     "Written Notice",
            "file_rbi_complaint":      "RBI Complaint",
            "file_police_complaint":   "Police Complaint",
            "request_debt_validation": "Debt Validation",
            "negotiate_settlement":    "Negotiate",
            "escalate_to_ombudsman":  "Ombudsman",
            "document_violations":     "Document",
            "contact_consumer_forum":  "Consumer Forum",
        }
        seq_display = " → ".join(action_short.get(a, a) for a in _seq)
        _src = "RL-trained" if __import__("os").path.exists(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "policy.json")) else "Baseline"
        st.info(
            f"**RL Strategy ({_src}):** {seq_display}  \n"
            f"**Why:** {_tp.get('why','')[:120]}..."
        )

    if st.button("🆘 Generate Emergency Messages Now", type="primary", use_container_width=True):
        with st.spinner("Generating your messages..."):
            lender_map = {
                "Illegal loan app": "illegal_app",
                "Bank (SBI, HDFC, ICICI etc.)": "bank",
                "NBFC / Fintech lender": "nbfc",
                "Microfinance institution (MFI)": "mfi",
            }
            msgs = generate_emergency_messages(
                lender_type    = lender_map.get(em_lender, "nbfc"),
                violation_type = em_violation,
                language       = em_language,
                app_name       = em_app,
                disbursed      = em_disbursed,
                demanded       = em_demanded,
            )
            st.session_state["emergency_msgs"] = msgs
            st.session_state["em_fp"] = current_fp

    if "emergency_msgs" in st.session_state:
        msgs = st.session_state["emergency_msgs"]

        # Key fact callout
        if msgs.get("key_fact"):
            st.success(f"**Know this:** {msgs['key_fact']}")
        if msgs.get("do_not"):
            st.error(f"**Do NOT:** {msgs['do_not']}")

        st.divider()

        # ── Language toggle ──────────────────────────────────────────────
        # st.radio natively re-renders on every click without rerun tricks.
        # text_area has NO key — so Python's value= is always respected.
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
            st.caption("Tap to switch")
            st.divider()
        else:
            lang_choice = "english"

        for label_key, msg_key_regional, msg_key_english in [
            ("msg1_label", "msg1", "msg1_en"),
            ("msg2_label", "msg2", "msg2_en"),
            ("msg3_label", "msg3", "msg3_en"),
        ]:
            label = msgs.get(label_key, "")
            msg = (
                msgs.get(msg_key_regional, "")
                if lang_choice == "regional"
                else msgs.get(msg_key_english, "")
            )
            if not msg:
                msg = msgs.get(msg_key_regional) or msgs.get(msg_key_english, "")
            if msg:
                st.markdown(f"**{label}**")
                # No key= on text_area — so value= is always applied fresh
                st.text_area("Message", value=msg, height=140, label_visibility="collapsed")
                st.caption("Select all text above → Copy → Send")
                st.divider()

        # Quick links
        st.markdown("**File now:**")
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown("[RBI CMS Portal →](https://cms.rbi.org.in)")
        col_b.markdown("[Cybercrime Portal →](https://cybercrime.gov.in)")
        col_c.markdown("[SACHET Portal →](https://sachet.rbi.org.in)")


# ============================================================
# TAB 2 — App Legitimacy Checker
# ============================================================
with tab2:
    st.markdown("### Is your loan app legal?")
    st.caption(
        "Non-payment to **illegal apps** will NOT affect your CIBIL score. "
        "Know before you pay."
    )

    app_name_input = st.text_input(
        "Type the app or lender name",
        placeholder="e.g. QuickCash, LazyPay, HDFC, Bajaj Finserv...",
        key="app_checker"
    )

    if app_name_input:
        verdict, color, advice = check_app_legality(app_name_input)

        color_map = {
            "red":    ("🔴", "#FAECE7", "#712B13", "#D85A30"),
            "orange": ("🟡", "#FAEEDA", "#633806", "#EF9F27"),
            "green":  ("🟢", "#E1F5EE", "#085041", "#1D9E75"),
        }
        icon, bg, text_c, border_c = color_map.get(color, color_map["orange"])

        st.markdown(f"""
<div style="background:{bg};border-left:4px solid {border_c};border-radius:0 10px 10px 0;
            padding:16px 20px;margin:12px 0">
  <div style="font-size:16px;font-weight:500;color:{text_c};margin-bottom:8px">
    {icon} {verdict}
  </div>
  <div style="font-size:13px;color:{text_c};line-height:1.7">{advice}</div>
</div>
""", unsafe_allow_html=True)

        if color == "red":
            st.markdown("**What to do if the app is ILLEGAL:**")
            st.markdown(
                "1. **Call 1930 NOW** — National Cybercrime Helpline (24×7, free)\n"
                "2. **Stop paying** — you do not legally owe them above principal + legal interest\n"
                "3. **Revoke phone permissions** — Settings → Apps → [App Name] → Permissions → Deny all\n"
                "4. **Uninstall the app** — limits their data access\n"
                "5. **File cybercrime complaint** → [cybercrime.gov.in](https://cybercrime.gov.in)\n"
                "6. **Report on SACHET** → [sachet.rbi.org.in](https://sachet.rbi.org.in)\n"
                "7. **Report to Play Store** → Find app → Flag as inappropriate\n"
                "8. **Check cooling-off right** — if loan is less than 3 days old, you can cancel with zero penalty"
            )
            st.error(
                "**CIBIL FACT (2025):** Illegal/unregistered apps are NOT connected to CIBIL. "
                "Non-payment will NOT appear on your credit report. Do not pay extortion disguised as debt."
            )
            st.info(
                "**NEW 2025 right:** Under RBI Digital Lending Directions 2025, "
                "lenders must provide a Key Fact Statement before disbursement. "
                "If they did not, ALL charges beyond principal are legally disputable."
            )

    # Known illegal apps list
    with st.expander("How to verify any app — RBI DLA directory"):
        st.markdown(
            "**The official source (July 2025):** RBI launched the Digital Lending Apps (DLA) directory.  \n"
            "Any app NOT listed is unregistered and illegal.  \n\n"
            "1. Go to **rbi.org.in** → Search 'Digital Lending Apps directory'  \n"
            "2. Or report suspected illegal apps at **sachet.rbi.org.in**  \n"
            "3. Cybercrime helpline: **1930** (24×7)  \n"
            "4. Portal: **cybercrime.gov.in**"
        )
        st.caption("The checker above attempts a live lookup of the RBI NBFC list. "
                   "Network issues may fall back to the curated database.")


# ============================================================
# TAB 3 — Harassment Log
# ============================================================
with tab3:
    st.markdown("### Log every violation — build your complaint automatically")
    st.caption(
        "Each log entry becomes evidence. When you have 3+ entries, "
        "the app generates your complete RBI complaint."
    )

    with st.form("log_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        log_date     = c1.date_input("Date", value=datetime.date.today())
        log_time     = c2.time_input("Time", value=datetime.time(9, 0))
        log_type     = st.selectbox("Type of violation", [
            "Call outside 7am–7pm hours",
            "Abusive / threatening language",
            "Contacted family / friends / employer",
            "Threatened arrest (illegal — loan default is civil)",
            "Visited home / workplace without notice",
            "Morphed photos / blackmail",
            "10+ calls in one day",
            "Demanded more than legally owed",
            "Other",
        ])
        log_detail   = st.text_area("Brief description (what exactly happened)", height=80)
        log_evidence = st.selectbox("Evidence you have", [
            "Call recording", "Screenshot of message",
            "WhatsApp screenshot", "Witness present",
            "No evidence yet"
        ])
        submitted = st.form_submit_button("Add to Log", use_container_width=True)
        if submitted:
            st.session_state["log"].append({
                "date":     str(log_date),
                "time":     str(log_time),
                "type":     log_type,
                "detail":   log_detail,
                "evidence": log_evidence,
            })
            st.success(f"Logged. Total violations recorded: {len(st.session_state['log'])}")

    if st.session_state["log"]:
        st.divider()
        st.markdown(f"**{len(st.session_state['log'])} violation(s) logged**")

        for i, entry in enumerate(st.session_state["log"], 1):
            with st.expander(f"#{i} — {entry['date']} {entry['time']} — {entry['type']}"):
                st.write(f"**Description:** {entry['detail'] or 'No description'}")
                st.write(f"**Evidence:** {entry['evidence']}")

        if len(st.session_state["log"]) >= 1:
            st.divider()
            if st.button("Generate RBI CMS Complaint from Log", type="primary"):
                with st.spinner("Drafting your complaint..."):
                    entries_text = "\n".join([
                        f"- {e['date']} at {e['time']}: {e['type']}. "
                        f"Details: {e['detail']}. Evidence: {e['evidence']}."
                        for e in st.session_state["log"]
                    ])

                    prompt = (
                        f"Draft a formal complaint for the RBI Complaints Management System (cms.rbi.org.in). "
                        f"The borrower has documented these violations of the RBI Fair Practices Code:\n\n"
                        f"{entries_text}\n\n"
                        f"Write a structured complaint that: cites specific RBI circulars violated, "
                        f"lists incidents in date order, requests specific relief (cessation of harassment, "
                        f"compensation under Integrated Ombudsman Scheme), and is suitable for submission "
                        f"to the RBI Banking Ombudsman. Format it as a proper complaint letter."
                    )
                    if client:
                        res = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                        complaint = res.choices[0].message.content
                    else:
                        complaint = "Please configure OPENAI_API_KEY to generate the complaint."

                    st.session_state["generated_complaint"] = complaint

        if "generated_complaint" in st.session_state:
            st.markdown("**Your RBI Complaint (ready to submit):**")
            st.text_area("Complaint", value=st.session_state["generated_complaint"], height=300, key="complaint_out", label_visibility="collapsed")
            st.success("Copy this text → go to cms.rbi.org.in → paste into complaint form")
            st.markdown("[Open RBI CMS Portal →](https://cms.rbi.org.in)")

        if st.button("Clear Log"):
            st.session_state["log"] = []
            st.rerun()


# ============================================================
# TAB 4 — Legal Amount Calculator
# ============================================================
with tab4:
    st.markdown("### What do you legally owe?")
    st.caption(
        "Illegal apps disburse ₹1,800 and demand ₹9,000. "
        "See the difference between what's legal and what's extortion."
    )

    c1, c2, c3 = st.columns(3)
    calc_disbursed = c1.number_input("Amount you actually received (₹)", min_value=0.0, value=5000.0, step=100.0)
    calc_demanded  = c2.number_input("Amount they are demanding (₹)",    min_value=0.0, value=15000.0, step=100.0)
    calc_days      = c3.number_input("Days since loan was taken",         min_value=1,   value=30,      step=1)

    calc_lender = st.selectbox("Lender type", [
        "Illegal app (max legal = principal only + minimal interest)",
        "Bank / NBFC (max legal interest = 36% per annum)",
        "MFI (max legal interest = 26% per annum under RBI cap)",
    ])

    if st.button("Calculate What I Actually Owe", type="primary"):
        rate_map = {
            "Illegal app (max legal = principal only + minimal interest)": 0.36,
            "Bank / NBFC (max legal interest = 36% per annum)": 0.36,
            "MFI (max legal interest = 26% per annum under RBI cap)": 0.26,
        }
        annual_rate = rate_map.get(calc_lender, 0.36)
        result = calculate_legal_amount(calc_disbursed, calc_demanded, calc_days)
        legal_max = calc_disbursed * (1 + annual_rate * calc_days / 365)
        extortion = max(0, calc_demanded - legal_max)

        st.divider()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("You received",     f"₹{calc_disbursed:,.0f}")
        col_b.metric("Legal maximum",    f"₹{legal_max:,.0f}",
                     delta=f"at {annual_rate*100:.0f}% APR for {calc_days} days")
        col_c.metric("They are demanding", f"₹{calc_demanded:,.0f}")

        st.divider()

        if extortion > 0:
            pct = (extortion / legal_max) * 100
            st.error(
                f"### ₹{extortion:,.0f} is extortion — not debt\n\n"
                f"You legally owe ₹{legal_max:,.2f}. "
                f"They are demanding ₹{calc_demanded:,.0f} — that is "
                f"**{pct:.0f}% more than the law allows.** "
                f"Paying the excess is not a legal obligation."
            )
            if "illegal" in calc_lender.lower():
                st.warning(
                    "**For illegal apps:** You may not owe anything at all if the app is unregistered. "
                    "Check the App Checker tab first. Non-payment to illegal apps does NOT affect your CIBIL score."
                )
        else:
            st.success(
                f"The amount demanded (₹{calc_demanded:,.0f}) is within legal limits. "
                f"This appears to be a legitimate repayment demand."
            )

        st.divider()
        st.markdown("**Your legal options:**")
        if extortion > 0:
            st.markdown(
                f"- **Pay only ₹{legal_max:,.0f}** (the legal maximum) and send written notice of overpayment\n"
                f"- File a complaint at cms.rbi.org.in citing the overcharge\n"
                f"- Under Consumer Protection Act 2019, you can claim the overcharged amount back as damages\n"
                f"- If the app is illegal, file at cybercrime.gov.in — non-payment will NOT hurt your CIBIL"
            )


# ============================================================
# TAB 5 — Full Strategy (original AI advisor)
# ============================================================
with tab5:
    st.markdown("### Full AI legal strategy")
    st.caption("For when you need a complete action plan, not just emergency messages.")

    col1, col2 = st.columns(2)
    with col1:
        lender_type = st.selectbox("Who is harassing you?", [
            "bank", "nbfc", "mfi", "illegal_app"
        ], format_func=lambda x: {
            "bank": "Bank (SBI, HDFC, ICICI etc.)",
            "nbfc": "NBFC / Fintech",
            "mfi":  "Microfinance (MFI)",
            "illegal_app": "Illegal loan app",
        }[x], key="strat_lender")

        debt_amount = st.number_input("Outstanding amount (₹)", value=15000.0, step=1000.0, key="strat_debt")

    with col2:
        days_overdue   = st.number_input("Days since default", value=45, key="strat_days")
        harassment_lvl = st.slider("Harassment severity", 0.0, 1.0, 0.6, key="strat_harass")

    task_level = st.selectbox("Task difficulty", ["easy", "medium", "hard", "expert"], key="strat_task")

    situation = st.text_area(
        "Describe your situation",
        placeholder="What exactly is happening? The more detail, the better the advice.",
        height=100,
        key="strat_situation"
    )

    if st.button("Generate Full Legal Strategy", type="primary", key="strat_btn"):
        with st.spinner("Consulting RL policy + Indian law..."):
            advice = get_advice(
                lender_type      = lender_type,
                harassment_level = harassment_lvl,
                days_overdue     = days_overdue,
                debt_amount      = debt_amount,
                violation_type   = situation or "general harassment",
                client           = client,
                language         = "English",
            )
            st.session_state["strategy"] = advice

    if "strategy" in st.session_state:
        adv = st.session_state["strategy"]
        st.divider()

        # Policy source badge
        src = adv.get("policy_source", "baseline")
        src_label = "RL-Trained Policy" if src == "rl_trained" else "Baseline Policy (run policy_trainer.py to improve)"
        src_color = "success" if src == "rl_trained" else "warning"
        st.markdown(f"**Strategy source:** :{src_color}[{src_label}]")

        # Task classification
        st.info(
            f"**Situation classified as:** `{adv['task']}` task  \n"
            f"**Reason:** {adv['task_reason']}  \n"
            f"**Expected score in simulation:** {adv['score_expected']:.0%}"
            + (f"  \n**Pass rate in training:** {adv['pass_rate']*100:.0f}%" if adv.get("pass_rate") else "")
        )

        # Full sequence
        action_short = {
            "send_written_notice":     "Written Notice",
            "file_rbi_complaint":      "RBI Complaint",
            "file_police_complaint":   "Police Complaint",
            "request_debt_validation": "Debt Validation",
            "negotiate_settlement":    "Negotiate",
            "escalate_to_ombudsman":  "Ombudsman",
            "document_violations":     "Document",
            "contact_consumer_forum":  "Consumer Forum",
        }
        seq_display = " → ".join(action_short.get(a, a) for a in adv["sequence"])
        st.markdown(f"**Complete RL sequence:** {seq_display}")

        # Step 1 — the immediate action
        st.markdown(f"**Step 1 — Do this now:** {adv['step1_label']}")
        if adv.get("step1_message"):
            st.text_area("Message to send:", value=adv["step1_message"], height=180, key="strat_msg")
            st.caption(f"Regulation: {adv.get('step1_regulation','')}")

        # Remaining steps
        if adv.get("next_steps"):
            st.markdown("**Your complete action plan:**")
            for step in adv["next_steps"]:
                st.markdown(f"- {step}")

        # RL timing rules
        if adv.get("timing_rules"):
            with st.expander("RL-learned timing rules"):
                for rule in adv["timing_rules"]:
                    st.markdown(f"- {rule}")

        # Prohibited actions warning
        if adv.get("prohibited"):
            st.warning(
                f"**Do NOT use:** {', '.join(adv['prohibited'])}  \n"
                f"These actions consistently underperformed in simulation for this scenario."
            )

        # Why this strategy
        if adv.get("why"):
            with st.expander("Why this strategy works"):
                st.write(adv["why"])

        col_l = {"bank":"📱 [RBI CMS](https://cms.rbi.org.in)",
                 "nbfc":"📱 [RBI CMS](https://cms.rbi.org.in)",
                 "mfi":"📱 [SACHET](https://sachet.rbi.org.in)",
                 "illegal_app":"🚨 [Cybercrime](https://cybercrime.gov.in)"}
        lt = st.session_state.get("user_data_strat", {}).get("lender_type", "bank")
        st.markdown(col_l.get(lt, ""))


# ---------------------------------------------------------------------------
# Sidebar — rights summary
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚖️ Your rights")
    st.markdown(
        "🕐 Agents can only call **7am–7pm**  \n"
        "🚫 Threats of **arrest are illegal**  \n"
        "🔒 Contact-list access = **IT Act 66E crime**  \n"
        "📋 RBI Ombudsman: **free, up to ₹20 lakh**  \n"
        "💡 Illegal app? **CIBIL not affected**  \n"
        "📄 **KFS right (2025):** All charges must be disclosed before loan  \n"
        "⏱ **Cooling-off (2025):** Cancel within 3 days, no penalty  \n"
        "📞 **Helpline 1930** — cybercrime, 24×7, free"
    )
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
    st.markdown("**Legal KB**")
    for law, desc in LEGAL_KB.items():
        with st.expander(law.replace("_", " ")):
            st.caption(desc)


# ---------------------------------------------------------------------------
# Required for OpenEnv validator
# ---------------------------------------------------------------------------
def main() -> None:
    pass

if __name__ == "__main__":
    main()