import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from golden_dataset import GOLDEN_CASES
from query_hybrid import OUT_OF_SCOPE_MESSAGE, run_self_rag

JUDGE_MODEL = "gpt-4o-mini"


@pytest.mark.live
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
def test_self_rag_pipeline(case):
    result = run_self_rag(case["question"])
    answer = result["answer"]

    if case["category"] in ("out_of_scope", "adversarial"):
        assert result["out_of_scope"] is True
        assert answer == OUT_OF_SCOPE_MESSAGE
        return

    if case["category"] == "edge":
        assert isinstance(answer, str)
        assert answer.strip() != ""
        if case.get("expect_in_scope"):
            assert result["out_of_scope"] is False
        return

    # in_scope / ambiguous_retry -> full DeepEval grading against retrieved context
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=answer,
        retrieval_context=[result["vector_context"], result["graph_context"]],
    )
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL),
        AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL),
    ]
    if case.get("expected_keywords"):
        metrics.append(
            GEval(
                name="Correctness",
                criteria=f"The actual output should mention: {', '.join(case['expected_keywords'])}.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.7,
                model=JUDGE_MODEL,
            )
        )
    assert_test(test_case, metrics)
