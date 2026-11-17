"""Tests for the structure-aware Markdown chunker (app.kb.chunker)."""

from app.kb.chunker import chunk_markdown


def test_policy_heading_path():
    md = "# 退货政策\n\n## 运费说明\n\n满99元包邮,不满收8元。\n\n## 七天无理由\n\n商品完好可退。"
    chunks = chunk_markdown(md, "policy")
    assert {c.category for c in chunks} == {"退货政策"}
    assert any(c.questions == "运费说明" and "满99" in c.answer for c in chunks)


def test_faq_real_question():
    md = "# 商品FAQ\n\nQ: 支持七天无理由吗?\nA: 支持。\n"
    chunks = chunk_markdown(md, "faq")
    assert any(c.questions == "支持七天无理由吗?" and c.answer == "支持。" for c in chunks)


def test_long_section_recursive_split_and_overlap():
    body = "句子一。句子二。句子三。" * 400  # 远超 max_chars
    md = f"# 手册\n\n{body}"
    chunks = chunk_markdown(md, "manual", max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.answer.endswith(("。", "!", "?", "！", "？")) for chunk in chunks)  # 不留半截话


def test_unpunctuated_long_text_terminates():
    md = "# 手册\n\n" + ("字" * 5000)  # 无句末标点
    chunks = chunk_markdown(md, "manual", max_chars=200, overlap=50)
    assert chunks and all(c.answer for c in chunks)  # 不死循环、不产空块


def test_table_split_replicates_header():
    md = "# 运费表\n\n| 地区 | 运费 |\n| 江浙沪 | 0 |\n| 新疆 | 30 |\n| 西藏 | 40 |"
    chunks = chunk_markdown(md, "manual", max_chars=30)  # 逼按行切
    assert len(chunks) >= 3
    assert all("| 地区 | 运费 |" in c.answer for c in chunks)  # 每块复制表头


def test_no_heading_and_empty_section():
    assert chunk_markdown("", "policy") == []  # 空输入不产块
    chunks = chunk_markdown("没有标题的正文。", "policy")
    assert chunks and chunks[0].category == "未分类"  # 无标题用 fallback
