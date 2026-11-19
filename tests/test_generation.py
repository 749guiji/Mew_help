"""Tests for the answer-coverage LLM judge (evaluation.generation)."""

from evaluation.generation import AnswerCoverageResult, judge_answer_coverage


class FakeStructuredRunnable:
    """Callable stand-in coerced to a ``RunnableLambda`` by ``|``."""

    def __init__(self, result):
        self._result = result

    def __call__(self, inp):
        return self._result


class FakeLLM:
    """Records ``method`` and returns a runnable yielding a canned result."""

    def __init__(self, result):
        self._result = result
        self.called_model = None
        self.called_method = None

    def with_structured_output(self, model, method=None):
        self.called_model = model
        self.called_method = method
        return FakeStructuredRunnable(self._result)


def test_judge_answer_coverage_returns_covered():
    llm = FakeLLM(AnswerCoverageResult(covered=[0, 2]))

    out = judge_answer_coverage(llm, "q", "a", ["p1", "p2", "p3"])

    assert isinstance(out, AnswerCoverageResult)
    assert out.covered == [0, 2]
    assert llm.called_model is AnswerCoverageResult
    assert llm.called_method == "function_calling"


def test_judge_answer_coverage_returns_none_on_fc_miss():
    llm = FakeLLM(None)

    out = judge_answer_coverage(llm, "q", "a", ["p1"])

    assert out is None
