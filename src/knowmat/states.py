"""
Definition of the shared state used by the KnowMat 2.0 LangGraph workflow.

The state is a TypedDict with optional fields that are progressively
populated as the pipeline advances.  Each node in the LangGraph will
receive a copy of the state and return a dictionary of updates.  The
graph merges these updates into the existing state.

Important fields
----------------

* ``pdf_path``: The path to the PDF file being processed.  Set at the
  beginning of the run.
* ``paper_text``: The extracted plain text of the paper with the
  references removed.
* ``sub_field``: The detected sub‑field of materials science (e.g.
  experimental, computational, simulation, machine learning, hybrid).
* ``updated_prompt``: The extraction prompt with adjustments suggested by
  the sub‑field detection and evaluation agents.
* ``latest_extracted_data``: The most recent extraction result as a
  JSON‑serialisable dictionary.
* ``run_results``: A list of dictionaries summarising each evaluation run.
  Each entry contains at least ``run_id``, ``confidence_score``,
  ``rationale``, ``suggested_prompt`` and ``extracted_data``.
* ``run_count``: The number of extraction/evaluation cycles that have
  occurred so far.  Used by the orchestrator to limit retries.
* ``max_runs``: The maximum number of extraction/evaluation cycles to
  perform.  Default is 3 and set by the orchestrator.
* ``needs_rerun``: Set by the evaluation agent to indicate whether another
  extraction is required.
* ``final_data``: The final aggregated extraction output produced by the
  manager agent.
* ``rationale``: A human‑readable summary of how the final result was
  obtained.
* ``flag``: Indicates whether a human review is recommended for this
  extraction.
"""

from typing import TypedDict, List, Optional, Dict, Any


class EvaluationRun(TypedDict, total=False):
    """Structure used to record the outcome of an individual evaluation run."""

    run_id: int
    confidence_score: float
    rationale: str
    missing_fields: Optional[List[str]]
    hallucinated_fields: Optional[List[str]]
    suggested_prompt: Optional[str]
    extracted_data: Dict[str, Any]


class KnowMatState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Initial inputs
    pdf_path: str
    output_dir: Optional[str]
    
    # PDF parsing results
    paper_text: str
    
    # Sub-field detection results
    sub_field: Optional[str]
    updated_prompt: Optional[str]
    
    # Extraction and evaluation results
    latest_extracted_data: Dict[str, Any]
    run_results: List[EvaluationRun]
    run_count: int
    max_runs: int
    needs_rerun: bool
    
    # Manager aggregation results
    aggregated_data: Optional[Dict[str, Any]]  # Stage 1: Merged data from all runs
    aggregation_notes: Optional[str]  # Stage 1: Merge strategy notes
    final_data: Optional[Dict[str, Any]]  # Stage 2: Validated and corrected data
    aggregation_rationale: Optional[str]
    human_review_guide: Optional[str]
    
    # Flagging agent results
    final_confidence_score: Optional[float]
    confidence_rationale: Optional[str]
    needs_human_review: Optional[bool]
    flag: bool