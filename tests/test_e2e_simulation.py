"""
End-to-end integration test: Upload → Extract → Simulate → Report.

Uses mock LLM throughout (USE_MOCK_LLM=true).
Tests the full pipeline with the 农夫山泉 sample document.
"""

from __future__ import annotations

import os
import pytest
from httpx import ASGITransport, AsyncClient

# Ensure mock mode
os.environ["USE_MOCK_LLM"] = "true"

from app.main import app


@pytest.fixture
def sample_md_content() -> bytes:
    """Minimal markdown content simulating a market analysis."""
    text = (
        "# Market Analysis\n\n"
        "## Overview\n\n"
        "The Chinese bottled water market exceeds 200 billion yuan. "
        "NongfuSpring holds about 26% market share, ranked first. "
        "CR Vanguard (Yibao) follows with about 21%.\n\n"
        "## Recent Developments\n\n"
        "In February 2024, NongfuSpring launched a 1-yuan green bottle "
        "pure water product, interpreted as a direct attack on Yibao's "
        "pure water territory, raising price war concerns.\n\n"
        "## Key Variables\n\n"
        "- Consumer upgrade vs downgrade trends\n"
        "- Channel cost changes\n"
        "- Policy and regulatory developments\n"
    )
    return text.encode("utf-8")


@pytest.mark.asyncio
async def test_full_pipeline(sample_md_content: bytes):
    """
    E2E: upload doc → extract world model → run simulation → get report.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # ---- Step 1: Upload document ----
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("nongfu.md", sample_md_content, "text/markdown")},
        )
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        doc = resp.json()
        doc_id = doc["document_id"]
        assert doc["char_count"] > 0

        # ---- Step 2: Extract world model ----
        resp = await client.post(
            f"/api/v1/documents/{doc_id}/extract",
            json={"question": "未来12个月农夫山泉市场份额会如何变化？"},
        )
        assert resp.status_code == 200, f"Extract failed: {resp.text}"
        extraction = resp.json()
        assert extraction["document_id"] == doc_id
        assert len(extraction["world_model"]["actors"]) > 0

        # ---- Step 3: Verify world model is registered for simulation ----
        # The extract endpoint should auto-save to DB
        from app import db as _db
        wm_check = await _db.get_world_model(doc_id)
        assert wm_check is not None, "World model not saved to DB"

        # ---- Step 4: Run simulation ----
        resp = await client.post(
            "/api/v1/simulations",
            json={
                "world_model_id": doc_id,
                "question": "未来12个月农夫山泉市场份额会如何变化？",
                "rounds": 2,
            },
        )
        assert resp.status_code == 202, f"Simulation create failed: {resp.text}"
        sim_summary = resp.json()
        sim_id = sim_summary["simulation_id"]
        assert sim_summary["status"] == "pending"

        # Background task needs to complete — poll until done
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0.1)
            resp = await client.get(f"/api/v1/simulations/{sim_id}")
            assert resp.status_code == 200
            sim_detail = resp.json()
            if sim_detail["status"] in ("completed", "failed"):
                break

        assert sim_detail["status"] == "completed", f"Simulation failed: {sim_detail.get('error')}"
        assert len(sim_detail["rounds"]) == 2
        assert sim_detail["report"] is not None
        assert sim_detail["report"]["confidence"] > 0

        # ---- Step 5: Get report directly ----
        resp = await client.get(f"/api/v1/simulations/{sim_id}/report")
        assert resp.status_code == 200
        report = resp.json()
        assert "prediction" in report
        assert "confidence" in report
        assert "key_factors" in report

        # ---- Step 6: Ask follow-up ----
        resp = await client.post(
            f"/api/v1/simulations/{sim_id}/ask",
            json={"question": "哪些因素最关键？"},
        )
        assert resp.status_code == 200
        ask_resp = resp.json()
        assert ask_resp["answer"] != ""


@pytest.mark.asyncio
async def test_pipeline_with_intervention(sample_md_content: bytes):
    """
    E2E with intervention: upload → extract → simulate with intervention → verify.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # Upload + extract
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("nongfu2.md", sample_md_content, "text/markdown")},
        )
        doc_id = resp.json()["document_id"]

        resp = await client.post(
            f"/api/v1/documents/{doc_id}/extract",
            json={"question": "市场份额预测"},
        )
        assert resp.status_code == 200

        # Simulate with intervention at round 1
        resp = await client.post(
            "/api/v1/simulations",
            json={
                "world_model_id": doc_id,
                "question": "市场份额预测",
                "rounds": 2,
                "interventions": [
                    {
                        "round": 1,
                        "event": "新水质国标出台",
                        "description": "国家市场监管总局发布新版饮用水标准，提高矿物质含量要求",
                    }
                ],
            },
        )
        assert resp.status_code == 202
        sim_id = resp.json()["simulation_id"]

        # Wait for completion
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0.1)
            resp = await client.get(f"/api/v1/simulations/{sim_id}")
            sim = resp.json()
            if sim["status"] in ("completed", "failed"):
                break

        assert sim["status"] == "completed"
        # Intervention should appear in round 1
        r1 = sim["rounds"][0]
        assert any(
            "INTERVENTION" in d for d in r1["narrator"]["new_developments"]
        ), "Intervention not found in round 1 narrator output"


@pytest.mark.asyncio
async def test_simulation_list(sample_md_content: bytes):
    """Verify the list endpoint returns created simulations."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # Upload + extract
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("nongfu3.md", sample_md_content, "text/markdown")},
        )
        doc_id = resp.json()["document_id"]

        await client.post(
            f"/api/v1/documents/{doc_id}/extract",
            json={"question": "test"},
        )

        # Create a simulation
        resp = await client.post(
            "/api/v1/simulations",
            json={
                "world_model_id": doc_id,
                "question": "test",
                "rounds": 1,
            },
        )
        assert resp.status_code == 202

        # List should include it
        resp = await client.get("/api/v1/simulations")
        assert resp.status_code == 200
        sims = resp.json()
        assert len(sims) >= 1
