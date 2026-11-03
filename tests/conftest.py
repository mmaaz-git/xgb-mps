import pytest
import xgboost as xgb


def pytest_addoption(parser):
    parser.addoption("--require-metal", action="store_true", help="Fail instead of skipping without GPU access")


@pytest.fixture
def metal(request):
    info = xgb.metal_info()
    if not info["built"] or not info["available"]:
        message = f"Metal execution unavailable: {info}"
        if request.config.getoption("--require-metal"):
            pytest.fail(message)
        pytest.skip(message)
    return info

