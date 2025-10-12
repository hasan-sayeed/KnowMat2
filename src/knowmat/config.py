"""
Global configuration and environment handling for KnowMat 2.0.

This module attempts to locate and load environment variables from a
``.env`` file if present.  It also ensures that critical secrets such as
``OPENAI_API_KEY`` and ``LANGCHAIN_API_KEY`` are set.  When these
variables are missing the code will prompt interactively for them at runtime.

In addition, the module sets environment variables to enable LangSmith
tracing.  Setting ``LANGCHAIN_TRACING_V2=true`` enables the v2 tracer and
``LANGCHAIN_PROJECT=KnowMat2`` names the project so that traces from
different runs are grouped together.
"""

import os
from dotenv import load_dotenv, find_dotenv


# Try to locate a .env file.  The search order is:
# 1) A .env in the current working directory
# 2) A path specified by KNOWMAT2_ENV_FILE
# 3) The first .env found upwards from cwd
_cwd_dotenv = os.path.join(os.getcwd(), ".env")
if os.path.isfile(_cwd_dotenv):
    _env_path = _cwd_dotenv
else:
    _env_path = os.getenv("KNOWMAT2_ENV_FILE", "")
    if not _env_path:
        _env_path = find_dotenv(usecwd=True) or ""

if _env_path:
    load_dotenv(_env_path, override=False)


def _set_env(var: str) -> None:
    """Prompt for an environment variable if it is not already set.

    Parameters
    ----------
    var: str
        Name of the environment variable to ensure.
    """
    if var not in os.environ:
        import getpass
        os.environ[var] = getpass.getpass(f"{var}: ")


# Ensure that API keys required by LangChain and OpenAI are present
_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")

# Enable LangSmith tracing by default for this project
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "KnowMat2")