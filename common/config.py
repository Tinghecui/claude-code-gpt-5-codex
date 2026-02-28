import os
from pathlib import Path

# We don't need to do `dotenv.load_dotenv()` - litellm does this for us upon import.

from common.utils import env_var_to_bool


WRITE_TRACES_TO_FILES = env_var_to_bool(os.getenv("WRITE_TRACES_TO_FILES"), "false")

# Streaming diagnostic logs (disabled by default)
ENABLE_STREAM_DIAGNOSTIC_LOGS = env_var_to_bool(os.getenv("ENABLE_STREAM_DIAGNOSTIC_LOGS"), "false")

try:
    STREAM_DIAGNOSTIC_SLOW_GAP_MS = int(os.getenv("STREAM_DIAGNOSTIC_SLOW_GAP_MS", "500"))
except (TypeError, ValueError):
    STREAM_DIAGNOSTIC_SLOW_GAP_MS = 500

TRACES_DIR = Path(__file__).parent.parent / ".traces"
