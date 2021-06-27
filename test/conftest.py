import pytest

@pytest.fixture
def mock_no_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)