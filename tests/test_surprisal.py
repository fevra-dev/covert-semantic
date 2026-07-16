from csd.surprisal import MockReferenceModel, ReferenceModel


def test_mock_surprisals_length_matches_tokens():
    m = MockReferenceModel()
    s = m.surprisals("alpha beta gamma")
    assert len(s) == 3
    assert all(x >= 0 for x in s)


def test_mock_stilted_text_has_higher_mean():
    m = MockReferenceModel()
    natural = m.mean_surprisal("the cat sat on the warm mat")
    stilted = m.mean_surprisal("the chromatic feline reposed upon the thermal textile")
    assert stilted > natural  # rarer tokens -> higher surprisal, the stego signature


def test_mock_is_deterministic():
    m = MockReferenceModel()
    assert m.surprisals("repeatable input") == m.surprisals("repeatable input")


def test_mock_token_stats_returns_surprisals_and_ranks():
    m = MockReferenceModel()
    surprisals, ranks = m.token_stats("the cat sat on the warm mat")
    assert len(surprisals) == len(ranks) == 7
    assert all(isinstance(r, int) and r >= 0 for r in ranks)


def test_mock_token_stats_common_words_rank_below_rare():
    m = MockReferenceModel()
    _, common = m.token_stats("the the the")
    _, rare = m.token_stats("chromatic feline reposed")
    assert max(common) < min(rare)  # common words -> lower rank (closer to argmax)


def test_base_token_stats_defaults_ranks_none():
    # a ReferenceModel that only implements surprisals() still works (ranks=None)
    class OnlySurprisal(MockReferenceModel):
        pass

    s, r = ReferenceModel.token_stats(OnlySurprisal(), "alpha beta")
    assert len(s) == 2 and r is None
