"""Tests for the knowledge material source list (app.kb.sources)."""

from pathlib import Path

from app.kb.sources import content_type_from_path, list_kb_materials


def test_content_type_rule():
    assert content_type_from_path(Path("product-faq.md")) == "faq"
    assert content_type_from_path(Path("return-policy.md")) == "policy"
    assert content_type_from_path(Path("退货政策.md")) == "policy"
    assert content_type_from_path(Path("aftersale-manual.md")) == "manual"
    assert content_type_from_path(Path("product-catalog.md")) == "manual"


def test_list_materials_reads_knowledge_dir(tmp_path):
    (tmp_path / "b-manual.md").write_text("hello", encoding="utf-8")
    (tmp_path / "a-faq.md").write_text("世界", encoding="utf-8")

    mats = list_kb_materials(directory=tmp_path)

    assert [m.name for m in mats] == ["a-faq.md", "b-manual.md"]  # sorted by name
    assert mats[0].content_type == "faq"
    assert mats[0].chars == 2  # "世界"
    assert mats[1].content_type == "manual"
    assert mats[1].chars == 5  # "hello"
