two flame agent  Predictive Fire Scenario Simulator

NASA Space Apps Challenge 2026 | Microgravity Combustion

two flame agent  is a full-stack, multi-agent AI platform designed to predict fire behavior in microgravity environments. It supports future missions including NASA Artemis, ISRO Gaganyaan, and the Bharatiya Antariksh Station (BAS). The platform combines a real-time 3D physics simulator, an interactive React dashboard, a physics-based Data Science engine, and a strict Retrieval-Augmented Generation (RAG) AI guard.

Architecture

The project is built on a modern, scalable 3-tier architecture:

Frontend (UI):
3D WebGL Simulator: A real-time environment where users can adjust Oxygen, Airflow, and Gravity to visualize the flame morphing from a buoyant Earth teardrop (1G) to a spherical orbital diffusion shell (0G).
React Dashboard: A sleek, aerospace-grade mission control interface featuring telemetry, digital twin sliders, and an AI chat panel.
Backend (FastAPI):
Acts as the central orchestrator. It routes API requests, manages the AI tool-calling loop, and serves the data to the frontend.
Super Intelligence Layer (AI & RAG):
Local LLM: Powered by Ollama (Qwen 2.5:7b).
Hallucination Guard: The AI is strictly constrained by a Retrieval-Augmented Generation (RAG) guard. It cannot answer from memory. It force-calls the Data Engine for quantitative math, then searches the local corpus for exact citations. If a citation cannot be found, the agent returns guard_passed: false and refuses to answer.
Data Science Engine:
A Python-based physics engine built by our Data Architect. It calculates Limiting Oxygen Concentration (LOC) and predicts ML outcomes by finding the closest experimental analog using Euclidean distance.
International Data Integration

two flame agent  is not limited to one agency. Our data layer integrates research from multiple space programs:

NASA: Saffire I-IV, Apollo 1, and FLEX droplet experiments.
ISRO: IMEx-2026 Gaganyaan and BAS mission profiles.
JAXA: FLARE solid material (NOMEX) 3D flame data.
How to Run Locally

1. Install Dependencies:

bash
pip install -r requirements.txt
2. Start the Local LLM (Ollama):
Ensure you have Ollama installed and the Qwen model pulled:

bash
ollama serve
ollama pull qwen2.5:7b
3. Start the FastAPI Backend:

bash
uvicorn main:app --host 0.0.0.0 --port 8000
4. Serve the Frontend:
Open a new terminal and run:

bash
python3 -m http.server 8080
5. Open the UI:
Navigate to http://localhost:8080/index.html in your browser.

Live Demo

Frontend UI: https://darshangr76.github.io/two-flames-agent/
(Note: The AI backend requires a local Python/Ollama environment to process queries. The live link demonstrates the UI, while the video demonstrates the full AI integration).

YouTube link 
https://youtu.be/WvCZvmkJltQ
