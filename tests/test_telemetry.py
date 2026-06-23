from api.telemetry import RequestRecord, telemetry


def test_telemetry_record_and_snapshot():
    t = telemetry.__class__()
    t.record(
        RequestRecord(
            ts=1.0,
            tier="balanced",
            connector_id="a@x",
            model_id="m",
            stages=["chat"],
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            latency_ms=50.0,
            passed=True,
        )
    )
    snap = t.snapshot()
    assert snap["requests_total"] == 1
    assert snap["output_tokens_total"] == 20
    recent = t.recent(5)
    assert recent[0]["connector_id"] == "a@x"
