from app.retrieval.hybrid import reciprocal_rank_fusion
import uuid


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
