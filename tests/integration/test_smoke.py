import pytest


@pytest.mark.slow
def test_real_training_smoke() -> None:
    pytest.skip("Run manually with a tiny local dataset and one epoch in Docker")
