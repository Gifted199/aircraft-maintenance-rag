"""The two phase 5 failures, as offline tests.

  I02  "B340-28-5209": retrieval found the right task but the model abstained.
  U04  "What is the torque for the A-200 nose wheel axle nut?": the model answered with the main
       wheel value, which the part number and torque checks cannot catch.
"""

import pytest

from aero_rag.generation.generator import SYSTEM_PROMPT
from aero_rag.guardrails.component import component_mismatch, mismatch_message
from aero_rag.guardrails.output import check_output
from aero_rag.guardrails.retrieval import identifier_sources
from aero_rag.schemas import LLMAnswer
from aero_rag.service import RagService
from tests.fakes import FakeGenerator, StubSearcher, faithful_answer, make_hit

A200_AXLE = "A-200_32-41-01_rev03"
A200_BRAKE = "A-200_32-42-03_rev02"
C75_SHIMMY = "C-75_32-21-02_rev01"
B340_PUMP = "B-340_28-21-05_rev03"
NOSE_QUESTION = "What is the torque for the A-200 nose wheel axle nut?"
MAIN_VALUE = "The final torque is 135 Nm (A-200, task 32-41-01, revision 3)."
CANNOT = LLMAnswer(status="cannot_answer", answer="Not in the sources.", steps=[], source_ids=[])


def service_for(generator, hits, aircraft_type=None):
    return RagService(
        StubSearcher(hits, aircraft_type), generator, cosine_threshold=0.35,
        max_question_chars=500,
    )  # fmt: skip


# ------------------------------------------------------------------ component check (U04)


@pytest.mark.parametrize(
    ("question", "doc_id", "expected"),
    [
        (NOSE_QUESTION, A200_AXLE, ("nose", "main")),
        ("A-200 nosewheel axle nut torque", A200_AXLE, ("nose", "main")),
        ("nose wheel brake wear limit on the A-200", A200_BRAKE, ("nose", "main")),
        ("What is the torque for the main wheel axle nut on the A-200?", A200_AXLE, None),
        ("Part number of the nose gear shimmy damper for the C-75?", C75_SHIMMY, None),
        ("Shimmy damper on the main gear of the C-75", C75_SHIMMY, ("main", "nose")),
        # No position word in the question: nothing to contradict.
        ("What is the torque for the A-200 axle nut?", A200_AXLE, None),
        # "maintenance" must not count as "main".
        ("nose wheel maintenance interval", "A-200_34-11-02_rev02", None),
    ],
)
def test_component_mismatch_compares_position_words_with_the_task_title(question, doc_id, expected):
    assert component_mismatch(question, make_hit(doc_id)) == expected


def test_component_mismatch_reads_the_title_not_the_task_text():
    # The main wheel task text mentions other components; only its title decides.
    hit = make_hit(A200_AXLE)
    assert "nose" not in hit.title.lower()
    assert component_mismatch("main wheel axle nut", hit) is None


def test_output_check_rejects_a_value_from_a_different_component():
    context = [make_hit(A200_AXLE)]
    answer = faithful_answer(context, text=MAIN_VALUE)
    assert check_output(answer, context) == []  # without the question, 135 Nm looks verified
    (problem,) = check_output(answer, context, NOSE_QUESTION)
    assert "main" in problem and "nose" in problem and "cannot_answer" in problem


def test_wrong_component_answer_is_never_returned_and_the_abstention_names_the_covered_one():
    generator = FakeGenerator(lambda q, c, p: faithful_answer(c, text=MAIN_VALUE))
    service = service_for(generator, [make_hit(A200_AXLE, 0.97, 0.72)], "A-200")
    response = service.answer(NOSE_QUESTION)

    assert response.status == "abstained" and response.answer.startswith("I don't know")
    assert "135" not in response.answer and response.citations == []
    assert "Main Wheel Axle Nut" in response.answer and "32-41-01" in response.answer
    assert len(generator.calls) == 2  # rejected, retried with the problem, rejected again
    assert any("nose" in p for p in generator.calls[1]["problems"])


def test_model_abstaining_on_a_wrong_component_gets_the_code_built_explanation():
    generator = FakeGenerator(CANNOT)
    service = service_for(generator, [make_hit(A200_AXLE, 0.97, 0.72)], "A-200")
    response = service.answer(NOSE_QUESTION)
    assert response.status == "abstained" and len(generator.calls) == 1
    expected = mismatch_message(NOSE_QUESTION, make_hit(A200_AXLE))
    assert response.answer == "I don't know. " + expected
    assert "Not in the sources" not in response.answer  # the model's own text is never shown


def test_right_component_question_is_unaffected():
    generator = FakeGenerator(lambda q, c, p: faithful_answer(c, text=MAIN_VALUE))
    service = service_for(generator, [make_hit(A200_AXLE, 0.97, 0.72)], "A-200")
    response = service.answer("What is the torque for the A-200 main wheel axle nut?")
    assert response.status == "answered" and len(generator.calls) == 1


# ------------------------------------------------------------------ bare identifiers (I02)


def test_identifier_sources_locates_part_and_task_numbers_in_the_context():
    context = [make_hit(A200_AXLE), make_hit(B340_PUMP)]
    assert identifier_sources("B340-28-5209", context) == {"B340-28-5209": [2]}
    assert identifier_sources("task 32-41-01", context) == {"32-41-01": [1]}
    assert identifier_sources("What is the brake torque?", context) == {}


def test_model_abstaining_on_a_matched_part_number_is_retried_with_its_location():
    context_hit = make_hit(B340_PUMP, 0.95, 0.47)
    answer = "B340-28-5209 is the O-ring, fuel boost pump (B-340, task 28-21-05, revision 3)."
    generator = FakeGenerator(CANNOT, lambda q, c, p: faithful_answer(c, text=answer))
    response = service_for(generator, [context_hit]).answer("B340-28-5209")

    assert response.status == "answered" and response.answer == answer
    assert len(generator.calls) == 2
    (problem,) = generator.calls[1]["problems"]
    assert "B340-28-5209 appears word for word in source 1" in problem
    assert response.citations[0].task_number == "28-21-05"


def test_identifier_retry_happens_once_then_abstains():
    generator = FakeGenerator(CANNOT, CANNOT)
    response = service_for(generator, [make_hit(B340_PUMP, 0.95, 0.47)]).answer("B340-28-5209")
    assert response.status == "abstained" and len(generator.calls) == 2


def test_model_abstaining_without_an_identifier_is_not_retried():
    generator = FakeGenerator(CANNOT)
    response = service_for(generator, [make_hit(A200_AXLE)], "A-200").answer(
        "What is the tyre pressure for the A-200 main wheel?"
    )
    assert response.status == "abstained" and len(generator.calls) == 1


def test_prompt_has_the_identifier_and_component_rules():
    assert "only an identifier asks what it is" in SYSTEM_PROMPT
    assert "exact component the question names" in SYSTEM_PROMPT
