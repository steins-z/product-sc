# MiroFish API Documentation

Base URL: `http://localhost:8000/api/v1`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Health Check

```
GET /health
```

**Response** `200`:
```json
{"status": "ok"}
```

---

## P0 — Document Ingestion & World Model Extraction

### 1. Upload Document

```
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

**Form field**: `file` — PDF, TXT, or MD file (≤20K characters recommended for MVP)

**Response** `200`:
```json
{
  "document_id": "d_a1b2c3",
  "filename": "nongfu_analysis.md",
  "content_type": "text/markdown",
  "text": "Full parsed text content...",
  "char_count": 5432
}
```

**Errors**: `400` unsupported type · `422` unreadable file

---

### 2. Get Document Chunks

```
GET /api/v1/documents/{document_id}/chunks
```

Returns overlapping text chunks with unique IDs for traceability.

**Response** `200`:
```json
{
  "document_id": "d_a1b2c3",
  "chunks": [
    {"chunk_id": "d_a1b2c3_chunk_0", "text": "...", "token_count": 487},
    {"chunk_id": "d_a1b2c3_chunk_1", "text": "...", "token_count": 512}
  ],
  "total_chunks": 4
}
```

**Errors**: `404` document not found

---

### 3. Extract World Model (Synchronous)

```
POST /api/v1/documents/{document_id}/extract
Content-Type: application/json

{"question": "未来12个月内农夫山泉市场份额会如何变化？"}
```

Blocks until LLM extraction completes.

**Response** `200`:
```json
{
  "document_id": "d_a1b2c3",
  "question": "...",
  "world_model": {
    "actors": [
      {"name": "农夫山泉", "role": "market_leader", "description": "...", "source_ref": ["d_a1b2c3_chunk_0"]}
    ],
    "relationships": [
      {"from": "农夫山泉", "to": "华润怡宝", "type": "competition", "description": "...", "source_ref": ["d_a1b2c3_chunk_1"]}
    ],
    "timeline": [
      {"date": "2024-02", "event": "价格战", "description": "...", "source_ref": ["d_a1b2c3_chunk_2"]}
    ],
    "variables": [
      {"name": "消费升级趋势", "current_value": "分化中", "description": "...", "source_ref": ["d_a1b2c3_chunk_0"]}
    ]
  }
}
```

---

### 4. Extract World Model (Asynchronous)

```
POST /api/v1/extract
Content-Type: application/json

{"document_id": "d_a1b2c3", "question": "..."}
```

Starts extraction in background. Poll with `GET /tasks/{task_id}`.

**Response** `202`:
```json
{"task_id": "task_x1y2z3", "status": "processing"}
```

---

### 5. Get Task Status

```
GET /api/v1/tasks/{task_id}
```

**Response** `200`:
```json
{
  "task_id": "task_x1y2z3",
  "status": "completed",
  "result": { "document_id": "...", "question": "...", "world_model": {...} }
}
```

**Status values**: `processing` → `completed` | `failed`

---

### 6. Get World Model

```
GET /api/v1/world-model/{document_id}
```

Returns the latest extracted world model.

**Errors**: `404` no world model for this document

---

### 7. Replace World Model

```
PUT /api/v1/world-model/{document_id}
Content-Type: application/json
```

Full replacement. Request body = complete `ExtractionResponse`.

---

### 8. Patch World Model

```
PATCH /api/v1/world-model/{document_id}
Content-Type: application/json

{
  "operations": [
    {"op": "add", "entity_type": "actors", "data": {"name": "监管机构", "role": "regulator", "description": "..."}},
    {"op": "remove", "entity_type": "actors", "target": "旧参与者"},
    {"op": "replace", "entity_type": "variables", "target": "品牌信任度", "data": {"current_value": "下降"}}
  ]
}
```

**Operations**: `add` / `remove` / `replace` on `actors` / `relationships` / `timeline` / `variables`.

---

## P1 — Multi-Agent Simulation

### 9. Create Simulation

```
POST /api/v1/simulations
Content-Type: application/json

{
  "world_model_id": "d_a1b2c3",
  "question": "未来72小时舆情如何发展？",
  "rounds": 3,
  "actors_override": null,
  "interventions": [
    {"round": 2, "event": "竞品负面营销", "description": "主要竞争对手发起针对性社交媒体攻势"}
  ]
}
```

Starts simulation in background (returns immediately).

**Parameters**:
- `world_model_id` (required) — Document ID with extracted world model
- `question` (required) — Prediction question
- `rounds` (optional, default 3, max 10) — Simulation rounds
- `actors_override` (optional) — Actor name subset; null = all
- `interventions` (optional) — Events to inject at specific rounds

**Response** `202`:
```json
{
  "simulation_id": "sim_abc123",
  "status": "pending",
  "question": "...",
  "current_round": 0,
  "total_rounds": 3
}
```

---

### 10. List Simulations

```
GET /api/v1/simulations
```

**Response** `200`: Array of `SimulationSummary`.

---

### 11. Get Simulation Detail

```
GET /api/v1/simulations/{simulation_id}
```

Full result including all rounds and report.

**Round structure**:
```json
{
  "round_number": 1,
  "narrator": {"scene": "...", "new_developments": ["...", "..."], "narrator_guidance": "..."},
  "actions": [
    {
      "actor_name": "农夫山泉",
      "action": "发布官方声明",
      "reasoning": "稳定品牌形象",
      "outcome": {"success": true, "impact": "部分消费者信心恢复", "side_effects": ["引发媒体进一步关注"]}
    }
  ],
  "adjudication": {
    "summary": "...",
    "variable_changes": {"舆情热度": "上升"},
    "relationship_changes": [],
    "continue_simulation": true
  },
  "world_state_snapshot": {...}
}
```

**Status values**: `pending` → `running` → `completed` | `failed`

---

### 12. Get Prediction Report

```
GET /api/v1/simulations/{simulation_id}/report
```

Available only when status = `completed`.

**Response** `200`:
```json
{
  "summary": "农夫山泉舆情在72小时内经历了升温、扩散、回落三个阶段...",
  "most_likely_path": "舆情在24小时内达到峰值后逐步回落...",
  "turning_points": ["媒体跟进报道导致舆情扩散", "官方声明发布后热度回落"],
  "key_variables": ["品牌信任度: 略有下降但趋于稳定", "舆情热度: 峰值后回落"],
  "risks": ["二次舆情爆发", "竞品趁机营销"],
  "opportunities": ["危机公关提升品牌透明度"],
  "recommended_actions": ["持续监测舆情", "准备后续深度声明", "主动邀请媒体考察水源地"],
  "evidence_refs": ["Round 1: 媒体跟进报道", "Round 2: 官方声明发布"]
}
```

**Errors**: `404` not found · `409` simulation not completed

---

### 13. Ask Follow-up Question

```
POST /api/v1/simulations/{simulation_id}/ask
Content-Type: application/json

{"question": "为什么第3轮品牌信任度会回升？"}
```

AI answers based on full simulation context.

**Response** `200`:
```json
{
  "answer": "第3轮品牌信任度回升主要因为...",
  "evidence": ["Round 2: 农夫山泉发布水源地检测报告", "Round 3: 媒体转向正面报道"]
}
```

**Errors**: `400` simulation not completed · `404` not found

---

## Error Format

All errors return:
```json
{"detail": "Human-readable error message"}
```

Common codes: `200` success · `202` accepted (async) · `400` bad request · `404` not found · `409` conflict · `422` validation · `500` internal error

---

## Typical End-to-End Flow

```
1. POST /documents/upload          → document_id
2. POST /extract                   → task_id
3. GET  /tasks/{task_id}           → poll until completed
4. GET  /world-model/{doc_id}      → review extracted model
5. PATCH /world-model/{doc_id}     → (optional) fix/add entities
6. POST /simulations               → simulation_id
7. GET  /simulations/{sim_id}      → poll until completed
8. GET  /simulations/{sim_id}/report → read prediction
9. POST /simulations/{sim_id}/ask  → ask follow-ups
```
