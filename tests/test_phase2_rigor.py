from experiments.phase2_rigor import run_generate, run_analyze, CONSTRUCTIONS


def test_generate_is_resumable_and_full(tmp_path):
    corpus = str(tmp_path / "c.jsonl")
    r1 = run_generate(n=4, out=corpus, mock=True)
    assert r1["written"] == 4 * 4 * 2          # 4 constructions x 4 x {stego,cover}
    r2 = run_generate(n=4, out=corpus, mock=True)
    assert r2["written"] == 0                   # all ids present -> resume writes nothing
    assert r2["total"] == 32


def test_analyze_mock_emits_full_verdict_block(tmp_path):
    corpus = str(tmp_path / "c.jsonl")
    run_generate(n=4, out=corpus, mock=True)
    res = run_analyze(corpus, mock=True, k=2, n_boot=50)
    for key in ("per_construction", "held_out", "deltas", "holm", "verdict",
                "size_ablation", "decodability", "primary_p_ref"):
        assert key in res
    assert set(res["size_ablation"]) == {"0.5B", "1.5B", "3B"}
    assert res["primary_p_ref"] == "1.5B"
    # the confirmatory family is 2 candidates x 4 constructions x 2 modes = 16 comparisons
    assert len(res["deltas"]) == 16
    assert res["holm"]["n"] == 16
    # decodability recorded for every construction (mock encoders round-trip -> 1.0)
    assert set(res["decodability"]) == {c.value for c in CONSTRUCTIONS}
    assert res["decodability"][CONSTRUCTIONS[0].value] == 1.0
    # every per-construction cell carries the 5 detectors with OOF arrays for the bootstrap
    cell = res["per_construction"][CONSTRUCTIONS[0].value]
    assert {"perplexity", "cross_perplexity", "distributional",
            "ensemble_max", "ensemble_stacked"} <= set(cell)
    assert "oof_scores" in cell["distributional"]
