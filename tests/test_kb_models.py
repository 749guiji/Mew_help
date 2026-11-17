from app.db.models import KnowledgeChunk, QaExtractionStaging


def test_knowledge_chunk_roundtrip(session):
    c = KnowledgeChunk(category="售后>运费", questions="运费怎么收", answer="满99包邮")
    session.add(c); session.commit()
    assert c.vectorize_status == "pending" and c.vector_id is None
    # 自引用指针:插两块后回填
    c2 = KnowledgeChunk(category="售后", questions="退货", answer="7天无理由")
    session.add(c2); session.commit()
    c.next_chunk_id = c2.id; c2.prev_chunk_id = c.id; session.commit()
    assert session.get(KnowledgeChunk, c.id).next_chunk_id == c2.id


def test_qa_staging_defaults(session):
    s = QaExtractionStaging(batch_no="b1", question="q", answer="a")
    session.add(s); session.commit()
    assert s.status == "extracted"
