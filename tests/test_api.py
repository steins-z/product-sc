"""Tests for Task 4 (extract API) and Task 5 (world model edit API).

Usage:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_TEXT = """\
农夫山泉成立于1996年，总部位于浙江杭州。
创始人钟睒睒持有约84%的股份。
怡宝是主要竞争对手。
"""

QUESTION = "农夫山泉能否保持市场第一？"


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _upload_and_extract() -> tuple[str, str]:
    """Upload a doc via one-shot extract, return (task_id, document_id)."""
    resp = client.post(
        "/api/v1/extract",
        files={"file": ("test.txt", SAMPLE_TEXT.encode(), "text/plain")},
        data={"question": QUESTION},
    )
    assert resp.status_code == 202, resp.text
    data = resp.json()
    return data["task_id"], None  # document_id not in task response directly


def _upload_doc() -> str:
    """Upload via /documents/upload and return document_id."""
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", SAMPLE_TEXT.encode(), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["document_id"]


def _extract_sync(document_id: str) -> dict:
    """Run sync extraction and return response."""
    resp = client.post(
        f"/api/v1/documents/{document_id}/extract",
        json={"question": QUESTION},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
#  Task 4: POST /extract + GET /tasks/{id}                                     #
# --------------------------------------------------------------------------- #


class TestTask4Extract:
    def test_one_shot_extract_returns_202(self):
        resp = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", SAMPLE_TEXT.encode(), "text/plain")},
            data={"question": QUESTION},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] in ("processing", "completed")

    def test_get_task_completed(self):
        resp = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", SAMPLE_TEXT.encode(), "text/plain")},
            data={"question": QUESTION},
        )
        task_id = resp.json()["task_id"]

        # With TestClient, background tasks run synchronously
        task_resp = client.get(f"/api/v1/tasks/{task_id}")
        assert task_resp.status_code == 200
        data = task_resp.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["world_model"]["actors"]

    def test_get_task_not_found(self):
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404

    def test_extract_bad_file_type(self):
        resp = client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
            data={"question": QUESTION},
        )
        assert resp.status_code == 400

    def test_extract_empty_file(self):
        resp = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"", "text/plain")},
            data={"question": QUESTION},
        )
        assert resp.status_code == 400


# --------------------------------------------------------------------------- #
#  Task 5: PUT + PATCH /world-model/{document_id}                              #
# --------------------------------------------------------------------------- #


class TestTask5WorldModel:
    def _setup_world_model(self) -> str:
        """Upload, extract, return document_id with a world model stored."""
        doc_id = _upload_doc()
        _extract_sync(doc_id)
        return doc_id

    def test_get_world_model(self):
        doc_id = self._setup_world_model()
        resp = client.get(f"/api/v1/world-model/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["world_model"]["actors"]

    def test_get_world_model_not_found(self):
        resp = client.get("/api/v1/world-model/nonexistent")
        assert resp.status_code == 404

    def test_put_full_replace(self):
        doc_id = self._setup_world_model()
        new_data = {
            "actors": [
                {
                    "name": "TestActor",
                    "role": "test",
                    "description": "A test actor",
                    "source_ref": ["chunk_0"],
                }
            ],
            "relationships": [],
            "timeline": [],
            "variables": [],
        }
        resp = client.put(f"/api/v1/world-model/{doc_id}", json=new_data)
        assert resp.status_code == 200
        wm = resp.json()["world_model"]
        assert len(wm["actors"]) == 1
        assert wm["actors"][0]["name"] == "TestActor"
        assert wm["relationships"] == []

    def test_patch_add_actor(self):
        doc_id = self._setup_world_model()
        original = client.get(f"/api/v1/world-model/{doc_id}").json()
        original_count = len(original["world_model"]["actors"])

        patch = {
            "operations": [
                {
                    "op": "add",
                    "path": "actors",
                    "value": {
                        "name": "NewActor",
                        "role": "new",
                        "description": "Newly added",
                        "source_ref": [],
                    },
                }
            ]
        }
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 200
        assert len(resp.json()["world_model"]["actors"]) == original_count + 1

    def test_patch_remove_actor(self):
        doc_id = self._setup_world_model()
        original = client.get(f"/api/v1/world-model/{doc_id}").json()
        original_count = len(original["world_model"]["actors"])

        patch = {"operations": [{"op": "remove", "path": "actors", "index": 0}]}
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 200
        assert len(resp.json()["world_model"]["actors"]) == original_count - 1

    def test_patch_replace_variable(self):
        doc_id = self._setup_world_model()
        patch = {
            "operations": [
                {
                    "op": "replace",
                    "path": "variables",
                    "index": 0,
                    "value": {
                        "name": "Updated Variable",
                        "current_value": "999",
                        "description": "Replaced",
                        "source_ref": [],
                    },
                }
            ]
        }
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 200
        assert resp.json()["world_model"]["variables"][0]["name"] == "Updated Variable"

    def test_patch_invalid_path(self):
        doc_id = self._setup_world_model()
        patch = {"operations": [{"op": "add", "path": "invalid_field", "value": {}}]}
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 400

    def test_patch_remove_out_of_range(self):
        doc_id = self._setup_world_model()
        patch = {"operations": [{"op": "remove", "path": "actors", "index": 999}]}
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 400

    def test_patch_unknown_op(self):
        doc_id = self._setup_world_model()
        patch = {"operations": [{"op": "delete", "path": "actors", "index": 0}]}
        resp = client.patch(f"/api/v1/world-model/{doc_id}", json=patch)
        assert resp.status_code == 400
