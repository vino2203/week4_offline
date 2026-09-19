import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]


def pytest_collection_modifyitems(config, items):
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return

    skip_live = pytest.mark.skip(reason=f"Missing required env vars: {', '.join(missing)}")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
