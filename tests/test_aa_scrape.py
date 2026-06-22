from connectors.import_aa import fetch_aa_public


def test_fetch_aa_public_parses_embedded_json():
    html = '{"intelligence_index_v4_1\\":33,"livecodebench\\":0.65,"tau2_bench_telecom\\":90.1}'
    # monkeypatch urlopen
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
