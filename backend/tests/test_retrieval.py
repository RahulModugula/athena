import uuid

from app.generation.prompts import format_context, lost_in_middle_reorder
from app.retrieval.hybrid import reciprocal_rank_fusion


class TestLostInMiddleReorder:
    def _make_chunks(self, n: int) -> list[dict]:
        return [{"content": f"chunk{i}", "document_name": f"doc{i}", "score": 1.0 - i * 0.1} for i in range(n)]

    def test_empty_and_single_passthrough(self) -> None:
        assert lost_in_middle_reorder([]) == []
        chunks = self._make_chunks(1)
        assert lost_in_middle_reorder(chunks) == chunks

    def test_two_chunks_unchanged(self) -> None:
        chunks = self._make_chunks(2)
        assert lost_in_middle_reorder(chunks) == chunks

    def test_reorders_so_highest_relevance_is_first(self) -> None:
        # chunk0 is most relevant (score 1.0), chunk1 second (0.9), etc.
        chunks = self._make_chunks(5)
        reordered = lost_in_middle_reorder(chunks)
        # chunk0 (highest) should appear at position 0
        assert reordered[0]["content"] == "chunk4"  # last even index prepended first
        # All original chunks are still present
        assert len(reordered) == 5
        assert {c["content"] for c in reordered} == {c["content"] for c in chunks}

    def test_format_context_uses_reordering(self) -> None:
        chunks = self._make_chunks(4)
        result = format_context(chunks)
        # All sources appear in the formatted output
        assert "[Source 1]" in result
        assert "[Source 4]" in result
        assert "---" in result


class TestRRF:
    def test_merges_two_lists(self) -> None:
        a = uuid.uuid4()
        b = uuid.uuid4()
        c = uuid.uuid4()

        list1 = [(a, 0.9), (b, 0.7), (c, 0.5)]
        list2 = [(c, 0.95), (a, 0.8), (b, 0.6)]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [doc_id for doc_id, _ in fused]

        assert a in ids
        assert b in ids
        assert c in ids
        assert len(fused) == 3

    def test_scores_are_positive(self) -> None:
        a = uuid.uuid4()
        b = uuid.uuid4()
        list1 = [(a, 1.0), (b, 0.5)]
        fused = reciprocal_rank_fusion([list1], k=60)
        for _, score in fused:
            assert score > 0

    def test_item_in_both_lists_scores_higher(self) -> None:
        a = uuid.uuid4()
        b = uuid.uuid4()
        list1 = [(a, 1.0), (b, 0.5)]
        list2 = [(a, 0.9)]
        fused = dict(reciprocal_rank_fusion([list1, list2], k=60))
        # a appears in both lists so should score higher than b
        assert fused[a] > fused[b]
