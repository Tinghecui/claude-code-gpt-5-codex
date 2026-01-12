import os
from pathlib import Path

# We don't need to do `dotenv.load_dotenv()` - litellm does this for us upon import.

from common.utils import env_var_to_bool


WRITE_TRACES_TO_FILES = env_var_to_bool(os.getenv("WRITE_TRACES_TO_FILES"), "false")
TRACES_DIR = Path(__file__).parent.parent / ".traces"
