"""Phase 2: chunking invariants and the ingestion job, offline with fakes."""

from collections import Counter

import pytest

from aero_rag.docgen.render import render
from aero_rag.ingestion import chunker
from aero_rag.ingestion.chunker import ChunkTooLargeError, build_chunks, header_line
from aero_rag.ingestion.index import index_body, pipeline_body
from aero_rag.ingestion.pipeline import IngestionError, mark_latest, run_ingestion
from aero_rag.ingestion.source import S3Source
from tests.fakes import FakeEmbedder, FakeOpenSearch, FakeS3
from tests.test_manuals import DOCS, SPEC

INDEX, PIPELINE, DIM = "test-index", "test-pipeline", 64


def ingest(s3=None, client=None, embedder=None, **kwargs):
    s3 = s3 or FakeS3.from_manuals()
    client = client or FakeOpenSearch()
    embedder = embedder or FakeEmbedder(DIM)
    stats = run_ingestion(
        S3Source(s3, "bucket", "manuals/"),
        embedder,
        client,
        index=INDEX,
        pipeline=PIPELINE,
        embedding_dim=DIM,
        **kwargs,
    )
    return stats, client, embedder


# ------------------------------------------------------------------ chunking


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_one_parent_and_two_children_with_deterministic_ids(doc):
    chunks = build_chunks(doc, True, "manuals/x.md")
    assert [c.chunk_type for c in chunks] == ["parent", "reference", "procedure"]
    assert [c.id for c in chunks] == [
        doc.doc_id,
        f"{doc.doc_id}#reference",
        f"{doc.doc_id}#procedure",
    ]
    assert {c.parent_id for c in chunks} == {doc.doc_id}
    assert build_chunks(doc, True, "manuals/x.md") == chunks


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_every_chunk_starts_with_the_one_line_header(doc):
    header = header_line(doc)
    for value in (doc.aircraft_type, doc.task_number, doc.title, f"Revision {doc.revision}"):
        assert value in header
    assert "\n" not in header
    for chunk in build_chunks(doc, True, "k"):
        assert chunk.text.split("\n")[0] == header


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_procedure_child_holds_every_step_with_its_warnings_unsplit(doc):
    _, reference, procedure = build_chunks(doc, True, "k")
    blocks = procedure.text.split("\n\n")[2:]  # after the header line and "Procedure:"
    assert len(blocks) == len(doc.steps)
    for step, block in zip(doc.steps, blocks, strict=True):
        expected = [f"Step {step.number}"]
        expected += [f"WARNING: {w}" for w in step.warnings]
        expected += [f"CAUTION: {c}" for c in step.cautions]
        expected.append(step.text)
        assert block.split("\n") == expected  # alerts are inside the step, before its text
    assert "WARNING:" not in reference.text and "Step 1\n" not in reference.text


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_reference_child_holds_every_part_number_and_torque(doc):
    parent, reference, _ = build_chunks(doc, True, "k")
    for part in doc.parts:
        assert f"P/N {part.part_number}" in reference.text
    for torque in doc.torques:
        assert f"{torque.fastener}: {torque.text}" in reference.text
    for tool in doc.tools:
        assert tool in reference.text
    assert doc.revision_date.isoformat() in reference.text


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.doc_id)
def test_parent_is_the_full_task(doc):
    parent, reference, procedure = build_chunks(doc, True, "k")
    header = header_line(doc)
    assert reference.text.removeprefix(header) in parent.text
    assert procedure.text.removeprefix(header) in parent.text


def test_an_oversize_procedure_stops_the_job_instead_of_being_split(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_EMBEDDING_TOKENS", 50)
    with pytest.raises(ChunkTooLargeError):
        build_chunks(DOCS[0], True, "k")


# ------------------------------------------------------------------ is_latest


def test_mark_latest_agrees_with_the_spec():
    expected = {doc.doc_id: is_latest for doc, is_latest in SPEC.documents()}
    assert mark_latest(DOCS) == expected
    assert Counter(expected.values()) == {True: 24, False: 7}


def test_latest_is_decided_per_aircraft_and_task_not_globally():
    flags = mark_latest(DOCS)
    assert flags["C-75_32-41-01_rev02"] is True  # B-340 has revision 5 of the same task number
    assert flags["A-200_32-41-01_rev02"] is False


# ------------------------------------------------------------------ the job


def test_ingestion_indexes_93_documents_and_embeds_only_children():
    stats, client, embedder = ingest()
    docs = client.indexes[INDEX]
    assert len(docs) == 93
    assert Counter(d["chunk_type"] for d in docs.values()) == {
        "parent": 31,
        "reference": 31,
        "procedure": 31,
    }
    assert len(embedder.embedded) == 62
    for source in docs.values():
        assert ("embedding" in source) == (source["chunk_type"] != "parent")
    assert Counter(d["is_latest"] for d in docs.values()) == {True: 72, False: 21}
    assert stats["documents"] == 31 and stats["latest"] == 24 and stats["superseded"] == 7
    assert stats["embedding_texts"] == 62 and stats["embedding_calls"] == 1
    assert PIPELINE in client.pipelines and client.refreshed == 1


def test_running_twice_updates_instead_of_duplicating():
    _, client, _ = ingest()
    first = {k: dict(v) for k, v in client.indexes[INDEX].items()}
    ingest(client=client)
    assert client.indexes[INDEX] == first


def test_dry_run_embeds_nothing_and_writes_nothing():
    stats, client, embedder = ingest(dry_run=True)
    assert embedder.embedded == [] and client.indexes == {} and client.pipelines == {}
    assert stats["embedding_texts"] == 62 and stats["estimated_tokens"] > 0


def test_a_new_revision_in_the_bucket_flips_the_old_one_to_not_latest():
    _, client, _ = ingest()
    old_id = "A-200_24-31-01_rev03"
    assert client.indexes[INDEX][old_id]["is_latest"] is True

    old = next(d for d in DOCS if d.doc_id == old_id)
    later = old.revision_date.replace(year=2027)
    newer = old.model_copy(update={"revision": 4, "supersedes": 3, "revision_date": later})
    s3 = FakeS3.from_manuals()
    s3.objects[f"manuals/{newer.filename}"] = render(newer)
    ingest(s3=s3, client=client)

    docs = client.indexes[INDEX]
    assert len(docs) == 96
    for suffix in ("", "#reference", "#procedure"):
        assert docs[old_id + suffix]["is_latest"] is False
        assert docs[newer.doc_id + suffix]["is_latest"] is True


def test_one_unparseable_file_stops_the_job_before_anything_is_written():
    s3 = FakeS3.from_manuals()
    s3.objects["manuals/broken.md"] = "# Not a manual\n"
    client, embedder = FakeOpenSearch(), FakeEmbedder(DIM)
    with pytest.raises(IngestionError, match="manuals/broken.md"):
        ingest(s3=s3, client=client, embedder=embedder)
    assert client.indexes == {} and embedder.embedded == []


def test_the_same_task_revision_twice_in_the_bucket_is_an_error():
    s3 = FakeS3.from_manuals()
    s3.objects["manuals/copy.md"] = s3.objects["manuals/A-200_24-31-01_rev03.md"]
    with pytest.raises(IngestionError, match="same task revision"):
        ingest(s3=s3)


def test_s3_source_reads_only_markdown_under_the_prefix_across_pages():
    s3 = FakeS3.from_manuals()
    s3.objects["other/ignored.md"] = "x"
    keys = [key for key, _ in S3Source(s3, "bucket", "manuals/").iter_documents()]
    assert len(keys) == 31 and keys == sorted(keys)
    assert all(k.startswith("manuals/") and k.endswith(".md") for k in keys)


def test_wrong_embedding_dimension_is_caught_before_indexing():
    client = FakeOpenSearch()
    with pytest.raises(IngestionError, match="dimension"):
        ingest(client=client, embedder=FakeEmbedder(dim=8))
    assert client.indexes == {}


# ------------------------------------------------------------------ index definition


def test_index_mapping_matches_the_design():
    body = index_body(1536)
    properties = body["mappings"]["properties"]
    assert body["settings"]["index"]["knn"] is True
    embedding = properties["embedding"]
    assert embedding["type"] == "knn_vector" and embedding["dimension"] == 1536
    assert embedding["method"]["name"] == "hnsw"
    assert embedding["method"]["space_type"] == "cosinesimil"
    assert properties["text"]["type"] == "text"
    for field in ("aircraft_type", "ata_chapter", "task_number", "chunk_type", "parent_id"):
        assert properties[field]["type"] == "keyword"
    assert properties["revision"]["type"] == "integer"
    assert properties["revision_date"]["type"] == "date"
    assert properties["is_latest"]["type"] == "boolean"


def test_every_indexed_field_is_in_the_strict_mapping():
    properties = set(index_body(DIM)["mappings"]["properties"])
    _, client, _ = ingest()
    for source in client.indexes[INDEX].values():
        assert set(source) <= properties


def test_search_pipeline_is_min_max_with_weighted_arithmetic_mean():
    processor = pipeline_body()["phase_results_processors"][0]["normalization-processor"]
    assert processor["normalization"]["technique"] == "min_max"
    assert processor["combination"]["technique"] == "arithmetic_mean"
    weights = processor["combination"]["parameters"]["weights"]
    assert len(weights) == 2 and sum(weights) == pytest.approx(1.0)
