"""
Integration test: Nongfu Spring (农夫山泉) end-to-end pipeline.

Runs: parse → chunk → extract (mock LLM by default)
Validates: all source_refs point to valid chunk_ids.

Usage:
    cd mirofish/
    python -m tests.test_nongfu
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.parser import parse_document
from app.services.chunker import chunk_text
from app.services.extractor import extract_world_model

# --------------------------------------------------------------------------- #
#  Sample document: Nongfu Spring business report                              #
# --------------------------------------------------------------------------- #

NONGFU_REPORT = """\
农夫山泉股份有限公司（Nongfu Spring Co., Ltd.）2024年度市场分析报告

一、公司概况

农夫山泉成立于1996年，总部位于浙江省杭州市，是中国领先的包装饮用水和饮料企业。公司创始人钟睒睒持有约84.4%的股份，于2020年9月8日在香港交易所上市（股票代码：09633.HK），IPO当日市值突破4000亿港元，钟睒睒一度成为亚洲首富。

公司拥有10大水源地，包括浙江千岛湖、吉林长白山、湖北丹江口、广东万绿湖等，均为天然水源。农夫山泉坚持"天然水"定位，这与竞争对手怡宝的"纯净水"形成鲜明对比。公司旗下产品线包括：天然水（核心）、东方树叶（无糖茶）、茶π、维他命水、NFC果汁、婴儿水等。

截至2024年，农夫山泉在中国包装饮用水市场占有率约为26.4%，连续十二年保持行业第一。公司2023年营收达426.7亿元人民币，同比增长28.4%，净利润120.8亿元，净利润率28.3%。

二、竞争格局

中国包装饮用水市场规模约为2300亿元人民币（2024年预估），年复合增长率约8%。主要竞争者包括：

1. 华润怡宝（C'estbon）：华润集团旗下，纯净水品类市占率第一（约20.9%）。2024年4月在港交所上市，募资超过70亿港元。怡宝与农夫山泉在渠道上高度重合，竞争激烈。

2. 百岁山（Ganten）：定位高端天然矿泉水，市占率约10.1%，近年增长迅速。以"水中贵族"为品牌标签，在一线城市表现强势。

3. 元气森林（Genki Forest）：新锐品牌，以无糖气泡水切入市场，目标年轻消费者。虽然主攻品类与农夫山泉不完全重合，但在无糖茶饮领域（东方树叶 vs 元气森林纤茶）形成直接竞争。元气森林2023年营收约80亿元。

4. 康师傅、统一等传统饮料巨头在低端市场仍有较大份额，但增长乏力。

三、2024年重大事件

2024年初，网络上出现大量关于"天然水vs纯净水"的争论，部分自媒体指控农夫山泉的天然水含有溴酸盐等问题。虽然后经检测证明农夫山泉产品符合国家标准，但此次风波对品牌形象造成短期冲击，2024年Q1线上销售同比下降约12%。

钟睒睒随后发表公开信，强调"我们只做天然水的搬运工"的品牌理念，并公布了详细的水质检测报告。事件在Q2逐步平息，销售回暖。

2024年6月，农夫山泉推出纯净水新品牌"绿瓶纯净水"，首次进入纯净水赛道，被视为对怡宝的正面回应。此举引发行业震动，分析师认为这标志着农夫山泉从"天然水专一主义"转向"全品类覆盖"战略。

2024年8月，公司宣布在云南投资20亿元新建生产基地，预计2026年投产，年产能100万吨。

四、关键变量分析

1. 消费升级趋势：中国消费者对健康饮品的需求持续上升，天然水/矿泉水增速高于纯净水。这有利于农夫山泉的核心定位。

2. 原材料成本：PET瓶坯价格2024年同比上涨约8%，运输成本受油价影响波动。农夫山泉水源地偏远，物流成本高于行业平均约15%。

3. 渠道竞争：线下渠道（便利店、超市、餐饮）仍占包装水销售的85%以上。农夫山泉拥有约240万个终端网点，行业最多。怡宝约180万个。渠道密度是核心护城河。

4. 新品类增长：东方树叶2023年营收突破100亿元，成为中国无糖茶第一品牌。无糖茶品类整体增速超过50%，是公司最重要的增长引擎。

5. 国际化布局：农夫山泉2024年开始试水东南亚市场（越南、泰国），但海外收入占比仍不足1%。

6. ESG与可持续发展：水源地保护和塑料回收成为投资者关注焦点。公司承诺2030年前实现包装100%可回收化。
"""

PREDICTION_QUESTION = "农夫山泉能否在2025年保持中国包装饮用水市场份额第一的位置？"


def main() -> None:
    print("=" * 70)
    print("MiroFish P0 Integration Test — 农夫山泉 (Nongfu Spring)")
    print("=" * 70)

    # --- Step 1: Parse ---
    print("\n📄 Step 1: Parsing document...")
    doc = parse_document("nongfu_spring_2024_report.txt", NONGFU_REPORT.encode("utf-8"))
    print(f"   document_id: {doc.document_id}")
    print(f"   content_type: {doc.content_type}")
    print(f"   char_count: {doc.char_count}")

    # --- Step 2: Chunk ---
    print("\n✂️  Step 2: Chunking...")
    chunks_resp = chunk_text(doc.text, doc.document_id)
    print(f"   total_chunks: {chunks_resp.total_chunks}")
    valid_chunk_ids = set()
    for chunk in chunks_resp.chunks:
        valid_chunk_ids.add(chunk.chunk_id)
        print(f"   - {chunk.chunk_id}: {chunk.token_count} tokens, {len(chunk.text)} chars")
        # Show first 60 chars
        preview = chunk.text[:60].replace("\n", " ")
        print(f"     Preview: {preview}...")

    # --- Step 3: Extract ---
    print("\n🧠 Step 3: Extracting world model...")
    result = extract_world_model(
        document_id=doc.document_id,
        question=PREDICTION_QUESTION,
        chunks=chunks_resp.chunks,
        use_mock=True,  # Always mock in test
    )
    print(f"   chunks_processed: {result.chunks_processed}")
    print(f"   question: {result.question}")

    wm = result.world_model
    print(f"\n   Actors ({len(wm.actors)}):")
    for a in wm.actors:
        print(f"     - {a.name} [{a.role}] refs={a.source_ref}")
    print(f"\n   Relationships ({len(wm.relationships)}):")
    for r in wm.relationships:
        print(f"     - {r.from_actor} → {r.to_actor} [{r.type}] refs={r.source_ref}")
    print(f"\n   Timeline ({len(wm.timeline)}):")
    for t in wm.timeline:
        print(f"     - [{t.date}] {t.event} refs={t.source_ref}")
    print(f"\n   Variables ({len(wm.variables)}):")
    for v in wm.variables:
        print(f"     - {v.name} = {v.current_value} refs={v.source_ref}")

    # --- Step 4: Validate source_refs ---
    print("\n✅ Step 4: Validating source_refs...")
    all_refs: list[str] = []
    invalid_refs: list[str] = []

    for actor in wm.actors:
        all_refs.extend(actor.source_ref)
    for rel in wm.relationships:
        all_refs.extend(rel.source_ref)
    for event in wm.timeline:
        all_refs.extend(event.source_ref)
    for var in wm.variables:
        all_refs.extend(var.source_ref)

    for ref in all_refs:
        if ref not in valid_chunk_ids:
            invalid_refs.append(ref)

    if invalid_refs:
        print(f"   ❌ FAIL: {len(invalid_refs)} invalid source_refs found:")
        for ref in invalid_refs:
            print(f"      - {ref}")
        print(f"   Valid chunk_ids: {sorted(valid_chunk_ids)}")
        sys.exit(1)
    else:
        print(f"   ✅ PASS: All {len(all_refs)} source_refs point to valid chunk_ids")

    # --- Print full JSON ---
    print("\n📋 Full World Model JSON:")
    print("-" * 70)
    # Use model_dump with by_alias to get "from"/"to" instead of "from_actor"/"to_actor"
    output = result.model_dump(by_alias=True)
    print(json.dumps(output, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("✅ All checks passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
