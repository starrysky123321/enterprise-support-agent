from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("ACCESS_TOKEN_SECRET_KEY", "test-access-secret")
os.environ.setdefault("REFRESH_TOKEN_SECRET_KEY", "test-refresh-secret")
os.environ.setdefault("RESET_PASSWORD_TOKEN_SECRET", "test-reset-secret")
os.environ.setdefault("VERIFICATION_TOKEN_SECRET", "test-verification-secret")
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "false")
os.environ.setdefault("EVALUATION_JUDGE_ENABLED", "false")
