# ZENITHFORGE: Predictive Fire Scenario Simulator

**NASA Space Apps Challenge 2026 | Microgravity Combustion**

ZENITHFORGE is a full-stack AI platform designed to predict fire behavior in microgravity environments, supporting missions like NASA Artemis, ISRO Gaganyaan, and JAXA. The system combines a 3D physics simulator, an AI research agent, and a retrieval-augmented generation (RAG) guard to ensure scientific claims are grounded in real experimental data.

## Architecture

1. Frontend: A WebGL 3D simulator and React dashboard providing an aerospace-grade user interface.
2. Backend: A FastAPI server that manages API requests and orchestrates the AI tools.
3. AI Layer: A local Ollama instance (Qwen2.5:7b) acting as a reasoning agent.
4. Data Engine: A Python engine that queries NASA, ISRO, and JAXA datasets to calculate Limiting Oxygen Concentration (LOC).

## The Hallucination Guard (RAG)

The AI agent is strictly constrained by a tool-calling loop. It does not answer from memory. It calls the simulator for math, then calls a local RAG search for citations. If a citation cannot be found, the agent returns a guard_passed value of false and refuses to answer.

## International Integration

- NASA: Saffire I-IV, Apollo 1, and FLEX droplet experiments.
- ISRO: IMEx-2026 Gaganyaan and Bharatiya Antariksh Station (BAS) profiles.
- JAXA: FLARE solid material (NOMEX) 3D flame data.

## How to Run Locally

1. pip install -r requirements.txt
2. ollama serve
3. uvicorn main:app --host 0.0.0.0 --port 8000
4. python3 -m http.server 8080
5. Open http://localhost:8080/zenithforge_combustionlab_nasa%20spaceapps.html

## Team ZENITHFORGE

- Member 1: Data Architect and Science Lead
- Member 2: AI/ML Engineer (RAG Guard and Agent Tools)
- Member 3: Frontend Architect
- Member 4 and 5: Storyteller and Product Manager
- Member 6: QA and Integration
