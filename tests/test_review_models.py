"""Tests for ch09 review_queue / eval_runs models + low_confidence_questions new columns."""

from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue


def test_review_queue_and_eval_run_models(session):
    rq = ReviewQueue(
        normalized_question="运费怎么算",
        ai_suggested_answer="满99包邮",
        occurrence_count=3,
    )
    session.add(rq)
    session.commit()
    session.refresh(rq)
    assert rq.review_status == "待审"
    assert rq.occurrence_count == 3
    assert rq.approved_answer is None

    er = EvalRun(triggered_by="手动", dataset_size=10, metrics={"recall_at_k": 0.8})
    session.add(er)
    session.commit()
    session.refresh(er)
    assert er.metrics["recall_at_k"] == 0.8


def test_low_confidence_question_new_columns(session):
    q = LowConfidenceQuestion(
        raw_question="运费?",
        source="retrieval_low_conf",
        retrieved_chunks=[{"score": 0.9, "answer": "满99包邮"}],
        matched_review_id=None,
    )
    session.add(q)
    session.commit()
    session.refresh(q)
    assert q.retrieved_chunks[0]["score"] == 0.9
    assert q.matched_review_id is None
