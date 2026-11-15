
"""
Two Flames in Free Fall — Backend (Final Complete Version)
FastAPI + local Ollama (qwen2.5:7b)

Endpoints: /predict, /sweep, /compare, /timeline, /ask, /agent, /version, /health
"""
import os
import re
import json
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- ZENITHFORGE DATA ENGINE INTEGRATION ---
from api_payloads.query_engine import inverse_query, predict_loc_bounded
from src.safety_insights import generate_safety_insight
from src.ml_predictor import classify_outcome


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
AGENT_MAX_ITER = int(os.getenv("AGENT_MAX_ITER", "5"))
REFUSAL = "I cannot answer based on NASA data."

app = FastAPI(title="Two Flames in Free Fall API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# NASA Corpus (Loaded from data/corpus/)
# ---------------------------------------------------------------------------
CORPUS_DIR = os.getenv("CORPUS_DIR", "data/corpus")
if os.path.isdir(CORPUS_DIR):
    _chunks = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(CORPUS_DIR, fname), "r") as f:
                _chunks.append(f.read())
    NASA_CORPUS = "\n\n".join(_chunks)
else:
    NASA_CORPUS = "ERROR: Corpus directory not found."

CORPUS_KEYWORDS = {
    "saffire", "flame", "fire", "oxygen", "o2", "microgravity", "gravity",
    "spherical", "teardrop", "spread", "spread rate", "flammability", "iss",
    "spacecraft", "combustion", "apollo", "bass", "flare", "uloc", "lunar",
    "moon", "material", "burn", "airflow", "flow", "cm/s", "cabin",
    "atmosphere", "nasa", "cygnus", "sibal", "cotton", "mylar",
    "quiescent", "buoyancy", "convection",
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    oxygen_percentage: float = Field(..., ge=0, le=100)
    airflow_m_s: float = Field(..., ge=0, le=10)
    gravity_level: str = Field("microgravity")

class PredictResponse(BaseModel):
    predicted_spread_rate: float
    predicted_spread_rate_ci95: list[float]
    uncertainty_reason: str
    flame_shape: str
    risk_level: str
    explanation: str

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class AskResponse(BaseModel):
    answer: str
    grounded: bool
    source: str

class Scenario(BaseModel):
    oxygen_percentage: float = Field(..., ge=0, le=100)
    airflow_m_s: float = Field(..., ge=0, le=10)
    gravity_level: str = Field("microgravity")

class AgentRequest(BaseModel):
    scenario: Scenario
    session_id: Optional[str] = None

class ToolStep(BaseModel):
    iteration: int
    tool: str
    args: dict
    result_summary: str

class NasaContextItem(BaseModel):
    claim: str
    full_text: str
    source: str

class AgentResponse(BaseModel):
    scenario: dict
    prediction: Optional[dict] = None
    nasa_context: list[NasaContextItem] = []
    verdict: str
    confidence: str
    guard_passed: bool
    steps: list[ToolStep] = []

class SweepRequest(BaseModel):
    airflow_m_s: float = Field(..., ge=0, le=10)
    gravity_level: str = Field("microgravity")

class CompareRequest(BaseModel):
    scenario_a: Scenario
    scenario_b: Scenario

class TimelineRequest(BaseModel):
    oxygen_percentage: float = Field(..., ge=0, le=100)
    airflow_m_s: float = Field(..., ge=0, le=10)
    gravity_level: str = Field("microgravity")
    horizon_seconds: int = Field(600, ge=10, le=3600)

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def _normalize_gravity(g: str) -> str:
    g = (g or "").strip().lower()
    if g in ("microgravity", "micro-g", "micro_g", "ug", "zero-g", "0g", "freefall", "free-fall"):
        return "microgravity"
    if g in ("lunar", "moon"):
        return "lunar"
    if g in ("mars", "martian"):
        return "mars"
    if g in ("earth", "1g", "terrestrial"):
        return "earth"
    if g in ("gaganyaan", "isro", "bas"):
        return "gaganyaan"
    return "microgravity"

def _predict(oxygen_pct: float, airflow_m_s: float, gravity: str) -> PredictResponse:
    g = _normalize_gravity(gravity)
    # Convert airflow m/s to cm/s for the data engine
    flow_cm_s = airflow_m_s * 100.0
    
    # Map our simple gravity to their mission profiles
    mission = "lunar" if g == "lunar" else ("mars" if g == "mars" else "custom")
    
    # Run their Inverse Query on NASA PSI data (Mocking NOMEX material for now)
    data_result = inverse_query("NOMEX", oxygen_pct, 101.3, flow_cm_s, mission)
    bounded = predict_loc_bounded("NOMEX", oxygen_pct, 101.3, flow_cm_s, mission)
    ml_outcome = classify_outcome("Heptane", 1.0, oxygen_pct/100.0, 3.0)
    
    confidence = data_result.get("confidence", "LOW")
    warning = data_result.get("warning", "")
    
    # Calculate a spread rate derived from their LOC data
    if bounded.get("status") == "SUCCESS":
        loc = bounded["predicted_LOC_pct"]
        # Higher LOC = further from extinction = faster spread
        spread = round(max(0.0, (oxygen_pct - loc) * 0.15), 3)
        if oxygen_pct < loc:
            spread = 0.0 # Extinguished
    else:
        spread = 0.0

    if oxygen_pct >= 30.0 or spread >= 1.0:
        risk = "High"
    elif oxygen_pct >= 24.0 or spread >= 0.3:
        risk = "Elevated"
    else:
        risk = "Low"

    if 20 <= oxygen_pct <= 35 and flow_cm_s <= 100.0:
        ci_spread = spread * 0.20
        uncertainty_reason = f"NASA PSI Data Engine: {confidence} Confidence."
    else:
        ci_spread = spread * 0.50
        uncertainty_reason = f"Data Gap: Extrapolated beyond NASA PSI validated envelope. {warning}"

    flame_shape = "Spherical" if flow_cm_s < 15 and g in ("microgravity", "lunar") else "Teardrop"
    
    safety_notes = generate_safety_insight("NOMEX", 1.0, oxygen_pct, "3D" if flame_shape == "Spherical" else "2D")
    safety_str = " ".join(safety_notes)

    explanation = (
        f"At {oxygen_pct:.1f}% O2, {airflow_m_s:.2f} m/s flow, {g} gravity. "
        f"NASA Data Engine predicts bounded LOC: {bounded.get('predicted_LOC_pct', 'N/A')}% "
        f"(Range: {bounded.get('lower_bound', 'N/A')}-{bounded.get('upper_bound', 'N/A')}%). "
        f"ML Outcome: {ml_outcome['prediction']} ({ml_outcome['reason']}). "
        f"Safety: {safety_str}"
    )

    return PredictResponse(
        predicted_spread_rate=spread,
        predicted_spread_rate_ci95=[round(max(0, spread - ci_spread), 3), round(spread + ci_spread, 3)],
        uncertainty_reason=uncertainty_reason,
        flame_shape=flame_shape, risk_level=risk,
        explanation=explanation.strip(),
    )

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
async def _ollama_generate(prompt: str, system: Optional[str] = None) -> str:
    if os.getenv("USE_MOCK_LLM") == "1":
        return "NASA data confirms the risk. Ensure oxygen stays below 30%."
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_predict": 512}}
    if system:
        payload["system"] = system
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama {r.status_code}: {r.text[:200]}")
    return (r.json().get("response") or "").strip()

async def _ollama_chat(messages: list, tools: Optional[list] = None) -> dict:
    if os.getenv("USE_MOCK_LLM") == "1":
        return {"role": "assistant", "content": "Mock response: high risk based on simulator."}
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0.0, "num_predict": 512}}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"Ollama {r.status_code}: {r.text[:200]}")
    return r.json().get("message", {})

def _question_in_corpus(q: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9%/]+", q.lower()))
    return bool(tokens & CORPUS_KEYWORDS)

# ---------------------------------------------------------------------------
# RAG logic
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_ASK = (
    "You are a strict NASA and ISRO document question-answering assistant. "
    "You may ONLY use the CONTEXT provided below. "
    "Do NOT use outside knowledge. Do NOT guess. "
    f"If the answer is not in the context, reply EXACTLY with: '{REFUSAL}' "
    "When you answer, you MUST cite the source in this exact format: "
    "'[NASA/TP-2015-216046, Page 51]' or '[ISRO IMEx-2026]'."
)

async def _fetch_live_ntrs(query: str) -> str:
    if os.getenv("USE_LIVE_NTRS") != "1":
        return ""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"https://ntrs.nasa.gov/api/citations/search?q={query}")
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])[:2]
                return " ".join([f"NASA NTRS: {res.get('title', '')} {res.get('abstract', '')}" for res in results])
    except Exception:
        pass
    return ""

async def _ask_logic(question: str) -> dict:
    if not _question_in_corpus(question):
        return {"answer": REFUSAL, "grounded": False, "source": "none"}

    live_data = await _fetch_live_ntrs(question)
    effective_corpus = NASA_CORPUS + ("\n\nLIVE NTRS EXTRACT:\n" + live_data if live_data else "")

    prompt = (
        f"NASA CONTEXT (ONLY source you may use):\n---BEGIN---\n{effective_corpus}\n---END---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using ONLY the context. If not present, reply exactly: '{REFUSAL}'"
    )
    answer = await _ollama_generate(prompt, system=SYSTEM_PROMPT_ASK)

    if answer.strip().lower().rstrip(".") == REFUSAL.lower().rstrip("."):
        retry_prompt = (
            f"NASA CONTEXT:\n---BEGIN---\n{effective_corpus}\n---END---\n\n"
            f"QUESTION: {question}\n\n"
            "Summarize what the NASA context says on this topic, even if it is not specific. "
            f"Only reply '{REFUSAL}' if completely absent."
        )
        answer = await _ollama_generate(retry_prompt, system=SYSTEM_PROMPT_ASK)
        if answer.strip().lower().rstrip(".") == REFUSAL.lower().rstrip("."):
            return {"answer": REFUSAL, "grounded": False, "source": "none"}

    answer_lower = answer.lower()
    grounded_markers = ("source", "saffire", "spherical", "teardrop",
                        "oxygen", "nasa", "microgravity", "apollo")
    if not any(m in answer_lower for m in grounded_markers):
        return {"answer": REFUSAL, "grounded": False, "source": "none"}

    m = re.search(r"\b(Saffire[-\s]?[IVX0-9]*|NASA[-\s]?STD[-\s]?6001|BASS|FLARE|ULOC|Apollo)\b",
                  answer, re.IGNORECASE)
    return {"answer": answer, "grounded": True,
            "source": m.group(0).strip() if m else "NASA corpus"}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    return AskResponse(**await _ask_logic(req.question.strip()))

# ---------------------------------------------------------------------------
# Simulator endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    return _predict(req.oxygen_percentage, req.airflow_m_s, req.gravity_level)

@app.post("/sweep")
async def sweep(req: SweepRequest):
    points = []
    for o2 in [15, 18, 21, 24, 27, 30, 33, 36, 40, 45]:
        p = _predict(o2, req.airflow_m_s, req.gravity_level)
        points.append({
            "oxygen_pct": o2,
            "spread_rate_mm_s": p.predicted_spread_rate,
            "flame_shape": p.flame_shape,
            "risk_level": p.risk_level,
        })
    return {"base": req.model_dump(), "curve": points}

@app.post("/compare")
async def compare(req: CompareRequest):
    pred_a = _predict(req.scenario_a.oxygen_percentage, req.scenario_a.airflow_m_s, req.scenario_a.gravity_level)
    pred_b = _predict(req.scenario_b.oxygen_percentage, req.scenario_b.airflow_m_s, req.scenario_b.gravity_level)
    diff_rate = round(pred_b.predicted_spread_rate - pred_a.predicted_spread_rate, 3)
    ratio = round(pred_b.predicted_spread_rate / pred_a.predicted_spread_rate, 2) if pred_a.predicted_spread_rate > 0 else "Infinity"
    return {
        "scenario_a": {"scenario": req.scenario_a.model_dump(), "prediction": pred_a.model_dump()},
        "scenario_b": {"scenario": req.scenario_b.model_dump(), "prediction": pred_b.model_dump()},
        "diff": {
            "absolute_increase_mm_s": diff_rate,
            "multiplier": f"{ratio}x",
            "risk_change": f"{pred_a.risk_level} -> {pred_b.risk_level}"
        }
    }

@app.post("/timeline")
async def timeline(req: TimelineRequest):
    pred = _predict(req.oxygen_percentage, req.airflow_m_s, req.gravity_level)
    rate_mm_s = pred.predicted_spread_rate
    points = []
    step = max(1, req.horizon_seconds // 10)
    for t in range(0, req.horizon_seconds + 1, step):
        points.append({
            "time_seconds": t,
            "cumulative_spread_mm": round(rate_mm_s * t, 2),
            "time_formatted": f"{t//60}m {t%60}s"
        })
    return {"scenario": req.model_dump(), "base_rate_mm_s": rate_mm_s, "curve": points}

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "predict_fire",
        "description": "Run the microgravity fire simulator.",
        "parameters": {"type": "object", "properties": {
            "oxygen_percentage": {"type": "number"},
            "airflow_m_s": {"type": "number"},
            "gravity_level": {"type": "string", "enum": ["microgravity", "lunar", "mars", "earth"]},
        }, "required": ["oxygen_percentage", "airflow_m_s", "gravity_level"]},
    }},
    {"type": "function", "function": {
        "name": "search_nasa",
        "description": "Look up NASA data on microgravity combustion.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}},
                       "required": ["question"]},
    }},
]

SYSTEM_PROMPT_AGENT = (
    "You are a NASA fire-safety reasoning agent for a lunar habitat. "
    "You MUST call predict_fire, then call search_nasa to cite NASA evidence. "
    "Only claim what the tools returned. Never invent numbers or sources. "
    "CRITICAL: Treat the simulator's flame_shape as ground truth. "
    "Do NOT contradict the simulator. "
    "If there is conversation history, explicitly compare the new scenario to the previous one. "
    "When done, write a 2-3 sentence verdict ending with a concrete mitigation recommendation."
)

_session_store = {}

def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}

async def _execute_tool(name: str, args: dict) -> dict:
    if name == "predict_fire":
        p = _predict(float(args.get("oxygen_percentage", 21)),
                     float(args.get("airflow_m_s", 0.0)),
                     str(args.get("gravity_level", "microgravity")))
        return p.model_dump()
    if name == "search_nasa":
        return await _ask_logic(str(args.get("question", "")))
    return {"error": f"unknown tool: {name}"}

def _summarize(tool: str, result: dict) -> str:
    if tool == "predict_fire":
        return (f"spread={result.get('predicted_spread_rate')} mm/s, "
                f"shape={result.get('flame_shape')}, risk={result.get('risk_level')}")
    if tool == "search_nasa":
        return f"grounded={result.get('grounded')}, source={result.get('source')}"
    return str(result)[:80]

@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest):
    session_id = req.session_id or "default"
    scenario = req.scenario.model_dump()
    past_messages = _session_store.get(session_id, [])
    steps: list[ToolStep] = []
    nasa_ctx: list[NasaContextItem] = []
    prediction: Optional[dict] = None
    predict_called = False
    search_called = False
    search_refused = False

    pred = await _execute_tool("predict_fire", scenario)
    prediction = pred
    predict_called = True
    steps.append(ToolStep(iteration=0, tool="predict_fire", args=scenario,
                          result_summary=_summarize("predict_fire", pred)))

    forced_q = ("What does NASA say about flame shape in microgravity and about "
                "oxygen concentration risks for spacecraft?")
    forced_search = await _execute_tool("search_nasa", {"question": forced_q})
    steps.append(ToolStep(iteration=0, tool="search_nasa",
                          args={"question": forced_q},
                          result_summary=_summarize("search_nasa", forced_search)))
    if forced_search.get("grounded"):
        search_called = True
        claim_text = (forced_search.get("answer") or "")[:300].strip()
        full_text = (forced_search.get("answer") or "").strip()
        src_tag = (forced_search.get("source") or "NASA corpus").strip()
        nasa_ctx.append(NasaContextItem(claim=claim_text, full_text=full_text, source=src_tag))
    else:
        search_refused = True

    messages = past_messages + [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": (
            f"Scenario: oxygen={scenario['oxygen_percentage']}%, "
            f"airflow={scenario['airflow_m_s']} m/s, gravity={scenario['gravity_level']}.\n"
            f"Simulator result: {json.dumps(pred)}\n\n"
            "Now call search_nasa with a GENERIC question about flame shape and O2 risk. "
            "Then write a short verdict."
        )},
    ]

    final_content = ""
    for iteration in range(1, AGENT_MAX_ITER + 1):
        msg = await _ollama_chat(messages, tools=TOOLS)
        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        if not tool_calls:
            final_content = content
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = _parse_args(fn.get("arguments"))
            result = await _execute_tool(name, args)
            steps.append(ToolStep(iteration=iteration, tool=name, args=args,
                                  result_summary=_summarize(name, result)))
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
            if name == "search_nasa":
                search_called = True
                if result.get("grounded"):
                    claim_text = (result.get("answer") or "")[:300].strip()
                    full_text = (result.get("answer") or "").strip()
                    src_tag = (result.get("source") or "NASA corpus").strip()
                    if not any(c.full_text == full_text and c.source == src_tag for c in nasa_ctx):
                        nasa_ctx.append(NasaContextItem(claim=claim_text, full_text=full_text, source=src_tag))
                else:
                    search_refused = True

    if not final_content:
        final_content = (f"Simulator reports spread {prediction['predicted_spread_rate']} mm/s "
                         f"({prediction['flame_shape']}, {prediction['risk_level']} risk).")

    guard_passed = predict_called and search_called and not search_refused and bool(nasa_ctx)
    o2 = scenario["oxygen_percentage"]
    flow = scenario["airflow_m_s"]
    if predict_called and search_called and not search_refused and 20 <= o2 <= 35 and flow <= 1.0:
        confidence = "high"
    elif predict_called and search_called:
        confidence = "medium"
    else:
        confidence = "low"
    verdict = final_content
    if not guard_passed:
        verdict = "[UNVERIFIED] " + verdict

    _session_store[session_id] = messages + [{"role": "assistant", "content": verdict}]

    return AgentResponse(
        scenario=scenario, prediction=prediction, nasa_context=nasa_ctx,
        verdict=verdict, confidence=confidence, guard_passed=guard_passed, steps=steps,
    )

# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------
@app.get("/version")
async def version():
    return {
        "app": "Two Flames in Free Fall", "version": "1.0.0",
        "model": OLLAMA_MODEL, "ollama_url": OLLAMA_URL,
        "agent_max_iter": AGENT_MAX_ITER,
    }

@app.get("/health")
async def health():
    ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            ok = r.status_code == 200
    except Exception:
        ok = False
    return {"status": "ok", "ollama_reachable": ok,
            "model": OLLAMA_MODEL, "ollama_url": OLLAMA_URL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
