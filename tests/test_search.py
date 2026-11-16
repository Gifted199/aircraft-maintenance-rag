"""Phase 2: search behaviour, offline. The index is filled by the real ingestion code via fakes."""

import pytest

from aero_rag.retrieval.aircraft import detect_aircraft, normalise_aircraft
from aero_rag.retrieval.search import Searcher, SearchMode
from tests.fakes import FakeEmbedder, FakeOpenSearch
from tests.test_ingestion import DIM, INDEX, PIPELINE, ingest

AXLE_Q = "What is the torque for the main wheel axle nut on the A-200?"


@pytest.fixture(scope="module")
def client() -> FakeOpenSearch:
    _, client, _ = ingest()
    return client


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(DIM)


@pytest.fixture
def searcher(client, embedder) -> Searcher:
    client.searches.clear()
    return Searcher(client, embedder, INDEX, PIPELINE)


# ------------------------------------------------------------------ aircraft detection


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (AXLE_Q, "A-200"),
        ("axle nut torque b340", "B-340"),
        ("C 75 battery removal", "C-75"),
        ("torque for the (c-75) pitot probe", "C-75"),
        ("A200-32-4112", None),  # a part number is not someone naming the aircraft
        ("install P/N B340-28-5209", None),
        ("main wheel axle nut torque", None),
        ("is the A-200 nut the same as the B-340 nut?", None),  # two aircraft: do not guess
        ("torque is 200 Nm", None),
        ("AA-200 or A-2000", None),
    ],
)
def test_detect_aircraft(text, expected):
    assert detect_aircraft(text) == expected


def test_normalise_aircraft_accepts_loose_input_and_rejects_unknown_types():
    assert normalise_aircraft(" a200 ") == "A-200"
    with pytest.raises(ValueError):
        normalise_aircraft("B-737")


# ------------------------------------------------------------------ the request we send


def sub_query_filters(body: dict) -> list[list[dict]]:
    keyword, knn = body["query"]["hybrid"]["queries"]
    return [keyword["bool"]["filter"], knn["knn"]["embedding"]["filter"]["bool"]["filter"]]


def test_hybrid_filters_are_inside_both_sub_queries_and_use_the_pipeline(searcher, client):
    searcher.search(AXLE_Q)
    call = client.searches[-1]
    assert call["params"] == {"search_pipeline": PIPELINE}
    assert "post_filter" not in call["body"]
    for filters in sub_query_filters(call["body"]):
        assert {"term": {"is_latest": True}} in filters
        assert {"term": {"aircraft_type": "A-200"}} in filters
        assert {"terms": {"chunk_type": ["reference", "procedure"]}} in filters


def test_include_superseded_is_the_only_way_to_drop_the_latest_filter(searcher, client):
    searcher.search(AXLE_Q, include_superseded=True)
    for filters in sub_query_filters(client.searches[-1]["body"]):
        assert not any("is_latest" in clause.get("term", {}) for clause in filters)
        assert {"term": {"aircraft_type": "A-200"}} in filters


def test_no_aircraft_means_no_aircraft_filter(searcher, client):
    searcher.search("main wheel axle nut torque")
    for filters in sub_query_filters(client.searches[-1]["body"]):
        assert not any("aircraft_type" in clause.get("term", {}) for clause in filters)
        assert {"term": {"is_latest": True}} in filters


def test_keyword_mode_makes_no_embedding_call_and_uses_no_pipeline(searcher, client, embedder):
    result = searcher.search(AXLE_Q, mode="keyword")
    call = client.searches[-1]
    assert embedder.embedded == [] and call["params"] == {}
    assert "bool" in call["body"]["query"]
    assert {"term": {"is_latest": True}} in call["body"]["query"]["bool"]["filter"]
    assert all(hit.cosine is None for hit in result.hits)


def test_vector_mode_filters_inside_the_knn_clause(searcher, client, embedder):
    result = searcher.search(AXLE_Q, mode=SearchMode.VECTOR)
    call = client.searches[-1]
    knn = call["body"]["query"]["knn"]["embedding"]
    assert call["params"] == {} and embedder.embedded == [AXLE_Q]
    assert {"term": {"is_latest": True}} in knn["filter"]["bool"]["filter"]
    assert all(hit.cosine is not None for hit in result.hits)


# ------------------------------------------------------------------ behaviour of the results


@pytest.mark.parametrize("mode", list(SearchMode))
def test_axle_nut_question_finds_the_latest_a200_task_first(searcher, mode):
    result = searcher.search(AXLE_Q, mode=mode)
    top = result.hits[0]
    assert (top.aircraft_type, top.task_number, top.revision) == ("A-200", "32-41-01", 3)
    assert result.aircraft_type == "A-200" and result.aircraft_source == "detected"
    assert result.aircraft_unspecified is False
    assert all(h.aircraft_type == "A-200" and h.is_latest for h in result.hits)


@pytest.mark.parametrize("mode", list(SearchMode))
def test_superseded_revisions_are_never_returned_by_default(searcher, mode):
    for query in (AXLE_Q, "cabin pressure sensor", "A200-32-4109", "battery terminal bolts 10 Nm"):
        assert all(hit.is_latest for hit in searcher.search(query, mode=mode, k=10).hits)


def test_given_aircraft_overrides_and_restricts_results(searcher):
    result = searcher.search("List the steps to replace the cabin pressure sensor", "b340")
    assert result.aircraft_type == "B-340" and result.aircraft_source == "given"
    assert result.hits[0].task_number == "21-31-04"
    assert {h.aircraft_type for h in result.hits} == {"B-340"}


def test_no_aircraft_searches_all_and_flags_it(searcher):
    result = searcher.search("main wheel axle nut torque")
    assert result.aircraft_unspecified is True and result.aircraft_type is None
    assert result.aircraft_source == "none"
    assert {h.aircraft_type for h in result.hits[:3]} == {"A-200", "B-340", "C-75"}
    assert {h.task_number for h in result.hits[:3]} == {"32-41-01"}


def test_part_number_query_finds_the_task_that_uses_it(searcher):
    result = searcher.search("A200-32-4112", mode="keyword")
    assert result.aircraft_unspecified is True
    assert result.hits[0].parent_id == "A-200_32-41-01_rev03"


def test_results_are_collapsed_to_distinct_parents_with_the_full_task_text(searcher, client):
    result = searcher.search(AXLE_Q, k=5)
    parent_ids = [h.parent_id for h in result.hits]
    assert len(parent_ids) == len(set(parent_ids)) == 5
    for hit in result.hits:
        assert hit.text == client.indexes[INDEX][hit.parent_id]["text"]
        assert "Torque values:" in hit.text and "Procedure:" in hit.text
        assert hit.matched_chunk_type in ("reference", "procedure")
    assert result.hits == sorted(result.hits, key=lambda h: -h.score)


def test_include_superseded_lets_old_revisions_appear(searcher):
    result = searcher.search(AXLE_Q, include_superseded=True, k=5)
    revisions = {h.revision for h in result.hits if h.task_number == "32-41-01"}
    assert revisions == {1, 2, 3}
    assert result.include_superseded is True
