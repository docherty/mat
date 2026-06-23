import httpx
import pytest

from connectors.provider_pricing import fetch_openrouter_pricing, fetch_venice_pricing


class _Resp:
    def __init__(self, payload: dict, *, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload


def test_fetch_openrouter_pricing_parses_prompt_completion(monkeypatch):
    payload = {
        "data": [
            {
                "id": "deepseek/deepseek-v4-flash",
                "pricing": {"prompt": "0.00000009", "completion": "0.00000018"},
            },
        ]
    }
    monkeypatch.setattr("connectors.provider_pricing.httpx.get", lambda *a, **k: _Resp(payload))
    out = fetch_openrouter_pricing("https://openrouter.ai/api/v1")
    p = out["deepseek/deepseek-v4-flash"].pricing
    assert p.input_per_1k == pytest.approx(0.00009)
    assert p.output_per_1k == pytest.approx(0.00018)


def test_fetch_venice_pricing_parses_model_spec_prices(monkeypatch):
    payload = {
        "object": "list",
        "data": [
            {
                "id": "xiaomi-mimo-v2-5",
                "model_spec": {"pricing": {"input": {"usd": 0.14}, "output": {"usd": 0.28}}},
            }
        ],
    }
    monkeypatch.setattr("connectors.provider_pricing.httpx.get", lambda *a, **k: _Resp(payload))
    out = fetch_venice_pricing("https://api.venice.ai/api/v1", api_key="k")
    p = out["xiaomi-mimo-v2-5"].pricing
    assert p.input_per_1k == pytest.approx(0.00014)
    assert p.output_per_1k == pytest.approx(0.00028)

