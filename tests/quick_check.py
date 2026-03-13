#!/usr/bin/env python3
"""Quick validation script."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Step 0: Imports...")
from app.services.parser import parse_document
from app.services.chunker import chunk_text
from app.services.extractor import extract_world_model
print("Imports OK")

sample = """农夫山泉成立于1996年，总部位于浙江杭州。创始人钟睒睒持有约84%股份。
公司2023年营收426.7亿元，净利润120.8亿元。
主要竞争对手包括怡宝、百岁山、元气森林等。
农夫山泉市占率约26.4%，连续十二年行业第一。"""

print("Step 1: Parse...")
doc = parse_document("test.txt", sample.encode("utf-8"))
print(f"  doc_id={doc.document_id}, chars={doc.char_count}")

print("Step 2: Chunk...")
chunks = chunk_text(doc.text, doc.document_id)
print(f"  total_chunks={chunks.total_chunks}")
for c in chunks.chunks:
    print(f"  {c.chunk_id}: {c.token_count} tokens")

print("Step 3: Extract...")
result = extract_world_model(doc.document_id, "农夫山泉能否保持第一?", chunks.chunks, use_mock=True)
print(f"  actors={len(result.world_model.actors)}")
print(f"  relationships={len(result.world_model.relationships)}")
print(f"  timeline={len(result.world_model.timeline)}")
print(f"  variables={len(result.world_model.variables)}")

print("Step 4: Validate refs...")
valid_ids = {c.chunk_id for c in chunks.chunks}
all_refs = []
for a in result.world_model.actors: all_refs.extend(a.source_ref)
for r in result.world_model.relationships: all_refs.extend(r.source_ref)
for t in result.world_model.timeline: all_refs.extend(t.source_ref)
for v in result.world_model.variables: all_refs.extend(v.source_ref)

bad = [r for r in all_refs if r not in valid_ids]
if bad:
    print(f"  FAIL: {len(bad)} invalid refs: {bad}")
    print(f"  Valid IDs: {sorted(valid_ids)}")
    sys.exit(1)
else:
    print(f"  PASS: all {len(all_refs)} refs valid")

print("\nALL CHECKS PASSED")
