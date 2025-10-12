"""
Application configuration for KnowMat 2.0.

This module defines a ``Settings`` class using pydantic's ``BaseSettings``
mechanism to manage environment‑configurable options such as the default
output directory, the OpenAI model to use and the generation temperature.

Environment variables are prefixed with ``KNOWMAT2_``.  For example,
``KNOWMAT2_OUTPUT_DIR`` overrides the default output directory and
``KNOWMAT2_MODEL_NAME`` changes the base model.  See the attributes of
``Settings`` for supported options.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration options for KnowMat 2.0.

    Attributes
    ----------
    output_dir: str
        Where extracted results and artifacts will be written.  Defaults to
        ``"data"`` relative to the current working directory.

    model_name: str
        The name of the ChatOpenAI model to use as fallback.  Defaults to ``"gpt-5"``.

    temperature: float
        Sampling temperature when generating with the language model.  A
        temperature of 0 yields deterministic outputs.  The default is 0.0.
        Note: GPT-5 models don't support custom temperature settings.
    
    subfield_model: str
        Model for subfield detection agent. Defaults to ``"gpt-5-mini"``.
    
    extraction_model: str
        Model for extraction agent. Defaults to ``"gpt-5"``.
    
    evaluation_model: str
        Model for evaluation agent. Defaults to ``"gpt-5"``.
    
    manager_model: str
        Model for validation agent (Stage 2: hallucination correction).
        Note: "manager_model" name kept for backward compatibility.
        Defaults to ``"gpt-5"``.
    
    flagging_model: str
        Model for flagging/quality assessment agent. Defaults to ``"gpt-5-mini"``.
    """

    output_dir: str = "data"
    model_name: str = "gpt-5"  # Fallback default
    temperature: float = 0.0  # Note: ignored for GPT-5 models
    
    # Per-agent model configuration
    subfield_model: str = "gpt-5-mini"
    extraction_model: str = "gpt-5"
    evaluation_model: str = "gpt-5"
    manager_model: str = "gpt-5"
    flagging_model: str = "gpt-5-mini"

    class Config:
        env_prefix = "KNOWMAT2_"


# Singleton instance to be imported throughout the package
settings = Settings()