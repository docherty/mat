from connectors.efficiency import token_efficiency_score


def test_token_efficiency_at_reference():
    assert token_efficiency_score(1000.0) == 1.0


def test_token_efficiency_double_reference():
    assert token_efficiency_score(2000.0) == 0.5
