from connectors.import_aa import _coherent_coding_index, _scrape_evaluations, fetch_aa_public


def test_coherent_coding_index_rejects_chart_leak():
    assert not _coherent_coding_index(4.9, 68.8)
    assert _coherent_coding_index(31.1, 43.4)
    assert _coherent_coding_index(33.0, 41.8)


def test_scrape_evaluations_skips_livecodebench_leak():
    html = (
        '{"livecodebench\\":0.878306878306878,'
        '"intelligence_index_v4_1\\":33,'
        '"coding_index\\":41.8838431127757,'
        '"gpqa\\":0.841414141414142}'
    )
    ev = _scrape_evaluations(html)
    assert "livecodebench" not in ev
    assert ev["artificial_analysis_coding_index"] == 41.8838431127757


def test_fetch_aa_public_parses_embedded_json():
    html = (
        '{"intelligence_index_v4_1\\":33,'
        '"coding_index\\":41.88,'
        '"tau2_bench_telecom\\":90.1}'
    )
    import connectors.import_aa as mod

    class FakeResp:
        def read(self):
            return html.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    mod.urlopen = lambda *a, **k: FakeResp()
    data = fetch_aa_public("qwen3-6-35b-a3b")
    assert data["evaluations"]["artificial_analysis_intelligence_index"] == 33.0
    assert data["evaluations"]["tau2_bench_telecom"] == 90.1
