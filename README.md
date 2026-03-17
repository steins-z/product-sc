# MiroFish — Multi-Agent Simulation Prediction Engine

> 把现实问题先在数字世界里跑一遍，再做决定。

MiroFish takes **seed materials** (news, reports, policy documents) + a **prediction question**, extracts a structured "world model", then runs **multi-agent simulations** to produce actionable prediction reports.

---

## Architecture

```
Materials (PDF/TXT/MD)
        │
        ▼
┌──────────────────┐
│  P0: Ingestion   │  Parse → Chunk → LLM Extract
│  Document → WM   │  actors / relationships / timeline / variables
└────────┬─────────┘
         │  WorldModel
         ▼
┌──────────────────┐
│  P1: Simulation  │  Narrator → Actor×N (parallel) → Adjudicator
│  Multi-Agent     │  Round-based, ≤10 rounds, ≤8 actors
└────────┬─────────┘
         │  SimulationResult
         ▼
┌──────────────────┐
│  Prediction      │  Summary · Path · Turning Points · Risks
│  Report          │  Opportunities · Recommendations · Evidence
└──────────────────┘
```

**P2: SQLite Persistence** — all documents, chunks, world models, tasks, and simulations are persisted to SQLite via aiosqlite. Survives restarts.

### Simulation Roles

| Role | Temp | Responsibility |
|------|------|----------------|
| **Narrator** | 0.3 | Sets the scene, advances timeline, injects interventions |
| **Actor** (×N) | 0.5 | Makes decisions from character's perspective (parallel via asyncio.gather) |
| **Adjudicator** | 0.1 | Resolves conflicts, updates variables, checks termination |
| **Report Generator** | 0.2 | Synthesizes all rounds into a structured prediction report |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/steins-z/product-sc.git
cd product-sc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY (DashScope key for Qwen, or OpenAI key)

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Interactive API docs
open http://localhost:8000/docs
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `qwen` | `qwen` or `openai` |
| `LLM_API_KEY` | — | DashScope API key (Qwen) or OpenAI API key |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `qwen-plus` | Model name |
| `USE_MOCK_LLM` | `true` | Set `false` for real LLM calls |
| `CHUNK_MAX_TOKENS` | `1000` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `100` | Overlap between chunks |
| `DB_FILENAME` | `mirofish.db` | SQLite database filename (under `data/`) |

---

## API Overview

Base URL: `http://localhost:8000/api/v1`

Full reference: [docs/API.md](docs/API.md)

### P0 — Document Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload PDF/TXT/MD → parsed text + document_id |
| `GET` | `/documents/{id}/chunks` | Get overlapping text chunks |
| `POST` | `/documents/{id}/extract` | Extract world model (sync) |
| `POST` | `/extract` | Extract world model (async, returns task_id) |
| `GET` | `/tasks/{task_id}` | Poll async task status |
| `GET` | `/world-model/{doc_id}` | Get extracted world model |
| `PUT` | `/world-model/{doc_id}` | Replace world model |
| `PATCH` | `/world-model/{doc_id}` | Incremental edits (add/remove/replace) |

### P1 — Simulation Engine

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulations` | Start simulation (async, 202) |
| `GET` | `/simulations` | List all simulations |
| `GET` | `/simulations/{id}` | Full simulation detail + rounds |
| `GET` | `/simulations/{id}/report` | Prediction report only |
| `POST` | `/simulations/{id}/ask` | Ask follow-up questions |

### Typical Workflow

```
Upload doc → Extract world model → (Review/Edit) → Run simulation → Read report → Ask follow-ups
```

---

## World Model Schema

```json
{
  "actors": [
    { "name": "农夫山泉", "role": "market_leader", "description": "...", "source_ref": ["chunk_0"] }
  ],
  "relationships": [
    { "from": "农夫山泉", "to": "华润怡宝", "type": "competition", "source_ref": ["chunk_1"] }
  ],
  "timeline": [
    { "date": "2024-02", "event": "价格战", "description": "...", "source_ref": ["chunk_2"] }
  ],
  "variables": [
    { "name": "品牌信任度", "current_value": "高", "description": "...", "source_ref": ["chunk_0"] }
  ]
}
```

All entities include `source_ref` arrays linking back to `chunk_id` for traceability.

---

## Testing

```bash
# All tests (mock LLM)
pytest

# Specific suites
pytest tests/test_api.py            # API endpoint tests
pytest tests/test_simulation.py     # Simulation engine tests
pytest tests/test_db.py             # Database persistence tests
pytest tests/test_e2e_simulation.py # End-to-end simulation

# Nongfu Spring integration test
python -m tests.test_nongfu

# With real Qwen API
USE_MOCK_LLM=false pytest tests/test_nongfu.py
```

---

## Project Structure

```
product-sc/
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   └── API.md                     # Full API reference
├── app/
│   ├── main.py                    # FastAPI app + lifespan (DB init)
│   ├── config.py                  # Settings (env-based)
│   ├── db.py                      # SQLite persistence (P2)
│   ├── models/
│   │   ├── document.py            # Document & chunk models
│   │   ├── world_model.py         # World model & extraction models
│   │   ├── task.py                # Async task tracking models
│   │   └── simulation.py          # Simulation engine models
│   ├── services/
│   │   ├── parser.py              # Document parsing (PDF/TXT/MD)
│   │   ├── chunker.py             # Text chunking with overlap
│   │   ├── extractor.py           # LLM extraction pipeline
│   │   └── simulation.py          # Multi-agent simulation engine
│   ├── api/
│   │   ├── routes.py              # P0 API routes
│   │   └── simulation_routes.py   # P1 simulation routes
│   └── prompts/
│       └── extraction.py          # Extraction prompt templates
├── data/                          # SQLite DB (auto-created at runtime)
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   └── nongfu_sample.md       # Sample: Nongfu Spring analysis
    ├── test_api.py                # API tests
    ├── test_db.py                 # DB persistence tests
    ├── test_simulation.py         # Simulation unit tests
    ├── test_e2e_simulation.py     # E2E simulation tests
    ├── test_nongfu.py             # Nongfu Spring integration
    └── quick_check.py             # Quick smoke test
```

---

## Development Milestones

- [x] **P0** — Document ingestion: parse → chunk → LLM extract → world model
- [x] **P1** — Simulation engine: Narrator + Actor + Adjudicator round loop
- [x] **P2** — SQLite persistence: all entities stored in DB, survives restarts
- [ ] **P3** — TBD

---

## License

Private — Steins-Z team.
