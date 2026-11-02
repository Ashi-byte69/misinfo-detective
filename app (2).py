"""
Misinformation Detective (FREE version using Google Gemini)
Pipeline: Extractor -> Searcher -> Fact-checker -> Judge
"""
import os
import json
import streamlit as st
from google import genai
from google.genai import types

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# The app tries these models in order and uses the first one that works.
# (gemini-2.5-flash was shut down by Google, which caused the 404 error)
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
]


# ---------- API key (works on your laptop and on Streamlit Cloud) ----------
def get_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


API_KEY = get_key()
if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Add it in Secrets (cloud) or as an environment variable (laptop).")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ---------- helper functions ----------
def llm(system: str, user: str) -> str:
    """One call to Gemini. Forces the answer to be valid JSON.
    If a model is not available, it automatically tries the next one."""
    last_error = None
    for model in MODELS:
        try:
            r = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                ),
            )
            return r.text
        except Exception as e:
            last_error = e
    raise last_error


def parse_json(text: str):
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ---------- Agent 1: Extractor ----------
def extractor(message: str) -> dict:
    system = (
        "You analyse viral messages. Reply ONLY with JSON in this form:\n"
        '{"claims": [{"claim": "...", "search_query": "..."}], '
        '"red_flags": ["..."]}\n'
        "claims = at most 3 checkable factual claims. "
        "search_query = a short web search query (3-8 words) to verify that claim. "
        "red_flags = manipulation tricks used (fake urgency, fear, 'forward to all', "
        "no source, miracle cure, authority name-dropping, etc). Can be empty."
    )
    return parse_json(llm(system, message))


# ---------- Agent 2: Searcher (a tool, no AI) ----------
def searcher(query: str) -> list:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]


# ---------- Agent 3: Fact-checker ----------
def fact_checker(claim: str, evidence: list) -> dict:
    system = (
        "You are a careful fact-checker. Use ONLY the evidence given. "
        "If the evidence is weak or missing, say UNVERIFIED - never guess. "
        "Reply ONLY with JSON:\n"
        '{"verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIED", '
        '"reason": "1-2 simple sentences", "sources": ["url", ...]}'
    )
    user = f"CLAIM: {claim}\n\nEVIDENCE:\n{json.dumps(evidence, indent=2)}"
    return parse_json(llm(system, user))


# ---------- Agent 4: Judge ----------
def judge(message: str, red_flags: list, checks: list) -> dict:
    system = (
        "You are the final judge. Given the original message, its red flags and "
        "the fact-check results, reply ONLY with JSON:\n"
        '{"overall": "LIKELY TRUE" | "LIKELY FALSE" | "MISLEADING" | "UNCERTAIN", '
        '"confidence": 0-100, '
        '"summary": "2-3 simple sentences", '
        '"reply_to_forward": "a polite, short message the user can send back to the '
        'person who forwarded it, in the SAME language as the original message"}'
    )
    user = json.dumps({"message": message, "red_flags": red_flags, "checks": checks}, indent=2)
    return parse_json(llm(system, user))


# ---------- the pipeline ----------
def run_pipeline(message: str, log):
    log("🕵️ Extractor: reading the message...")
    ext = extractor(message)
    claims, red_flags = ext.get("claims", []), ext.get("red_flags", [])
    log(f"   found {len(claims)} claim(s), {len(red_flags)} red flag(s)")

    checks = []
    for c in claims:
        log(f"🔎 Searcher: '{c['search_query']}'")
        evidence = searcher(c["search_query"])
        log(f"⚖️ Fact-checker: checking '{c['claim'][:60]}...'")
        result = fact_checker(c["claim"], evidence)
        result["claim"] = c["claim"]
        checks.append(result)

    log("👩‍⚖️ Judge: making the final decision...")
    final = judge(message, red_flags, checks)
    return red_flags, checks, final


# ---------- UI ----------
st.set_page_config(page_title="Misinformation Detective", page_icon="🕵️")
st.title("🕵️ Misinformation Detective")
st.caption("Paste a suspicious WhatsApp forward. Four AI agents will investigate it.")

message = st.text_area("Forwarded message", height=200)

if st.button("Investigate", type="primary") and message.strip():
    with st.status("Agents working...", expanded=True) as status:
        try:
            red_flags, checks, final = run_pipeline(message, st.write)
            status.update(label="Investigation complete", state="complete")
        except Exception as e:
            status.update(label="Something failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

    colour = {"LIKELY TRUE": "green", "LIKELY FALSE": "red",
              "MISLEADING": "orange", "UNCERTAIN": "gray"}.get(final["overall"], "gray")
    st.markdown(f"## Verdict: :{colour}[{final['overall']}]  ({final['confidence']}% confident)")
    st.write(final["summary"])

    if red_flags:
        st.subheader("🚩 Manipulation red flags")
        for f in red_flags:
            st.write(f"- {f}")

    st.subheader("Claim-by-claim")
    for c in checks:
        with st.expander(f"{c['verdict']} — {c['claim']}"):
            st.write(c["reason"])
            for s in c.get("sources", []):
                st.write(f"- {s}")

    st.subheader("💬 Reply you can send back")
    st.code(final["reply_to_forward"], language=None)
