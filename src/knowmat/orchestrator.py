"""
Top‑level assembly of the KnowMat 2.0 LangGraph workflow.

This module defines two functions: :func:`build_graph` constructs the
LangGraph state machine wiring the individual nodes together, and
:func:`run` orchestrates the end‑to‑end extraction process for a single
PDF document.  The flow is as follows:

1. The PDF is parsed into text.
2. The sub‑field detection agent analyses the text and updates the prompt.
3. An extraction is performed according to the current prompt.
4. The evaluation agent scores the extraction and determines whether another
   cycle is necessary.  Up to ``max_runs`` (default 3) cycles are
   attempted.
5. The manager agent aggregates all runs and produces the final output.

The run function can be invoked from the command line via ``python -m
knowmat2``.  It handles environment and settings initialisation similarly
to the MI‑Agent project and writes the final extraction to the specified
output directory.
"""

import os
import json
import textwrap
import uuid
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from knowmat.states import KnowMatState
from knowmat.post_processing import PostProcessor
from knowmat.nodes.docling_parse_pdf import parse_pdf_with_docling
from knowmat.nodes.subfield_detection import detect_sub_field
from knowmat.nodes.extraction import extract_data
from knowmat.nodes.evaluation import evaluate_data
from knowmat.nodes.aggregator import aggregate_runs  # Stage 1: Simple data merging
from knowmat.nodes.validator import validate_and_correct  # Stage 2: Hallucination correction & validation
from knowmat.nodes.flagging import assess_final_quality
from knowmat.app_config import Settings, settings
from knowmat.config import _env_path


def evaluation_condition(state: KnowMatState) -> str:
    """Decide whether to rerun extraction or proceed to aggregation.

    Called by the graph when the evaluation node completes.  If
    ``state['needs_rerun']`` is true and ``state['run_count']`` is less
    than ``state['max_runs']`` the function returns the name of the
    extraction node to trigger another cycle.  Otherwise it returns
    ``aggregate_runs`` to begin the two-stage manager process.
    """
    run_count = state.get("run_count", 0)
    max_runs = state.get("max_runs", 3)
    needs_rerun = state.get("needs_rerun", False)
    if needs_rerun and run_count < max_runs:
        return "extract_data"
    return "aggregate_runs"


def build_graph() -> StateGraph:
    """Construct the LangGraph for KnowMat 2.0 with two-stage manager."""
    builder = StateGraph(KnowMatState)
    
    # Register all nodes
    builder.add_node("parse_pdf", parse_pdf_with_docling)
    builder.add_node("detect_sub_field", detect_sub_field)
    builder.add_node("extract_data", extract_data)
    builder.add_node("evaluate_data", evaluate_data)
    builder.add_node("aggregate_runs", aggregate_runs)  # Stage 1: Merge runs
    builder.add_node("validate_and_correct", validate_and_correct)  # Stage 2: Correct hallucinations
    builder.add_node("assess_final_quality", assess_final_quality)
    
    # Wire edges - New two-stage manager flow
    builder.add_edge(START, "parse_pdf")
    builder.add_edge("parse_pdf", "detect_sub_field")
    builder.add_edge("detect_sub_field", "extract_data")
    builder.add_edge("extract_data", "evaluate_data")
    builder.add_conditional_edges("evaluate_data", evaluation_condition, ["extract_data", "aggregate_runs"])
    builder.add_edge("aggregate_runs", "validate_and_correct")  # Stage 1 → Stage 2
    builder.add_edge("validate_and_correct", "assess_final_quality")  # Stage 2 → Flagging
    builder.add_edge("assess_final_quality", END)
    
    return builder.compile(checkpointer=MemorySaver())


# def run(
#     pdf_path: str,
#     output_dir: Optional[str] = None,
#     model_name: Optional[str] = None,
#     max_runs: int = 3,
# ) -> dict:
#     """Run the full KnowMat 2.0 pipeline on a given PDF and write results.

#     Parameters
#     ----------
#     pdf_path: str
#         Path to the materials science paper in PDF format.
#     output_dir: Optional[str
#         Directory where the resulting JSON file will be saved.  If not
#         provided, the default from :mod:`knowmat2.app_config` is used.
#     model_name: Optional[str]
#         Override the base model used for extraction (e.g. ``"gpt-4"``).
#     max_runs: int
#         Maximum number of extraction/evaluation cycles to perform.  Defaults
#         to 3.

#     Returns
#     -------
#     dict
#         A dictionary containing the final aggregated extraction, the
#         rationale and the flag indicating whether human review is needed.
#     """
#     # Load environment variables if a .env was found
#     if _env_path:
#         print(f"🔍 Loaded environment variables from: {_env_path}")
#     # Apply any CLI overrides to settings
#     overrides = {}
#     if output_dir:
#         overrides["output_dir"] = output_dir
#     if model_name:
#         overrides["model_name"] = model_name
#     if overrides:
#         # Create a new Settings object and update the global settings
#         new_settings = Settings(**overrides)
#         settings.__dict__.update(new_settings.model_dump())
#     # Prepare the initial state
#     state: KnowMatState = {
#         "pdf_path": pdf_path,
#         "output_dir": output_dir,  # Add output_dir to state for docling parser
#         "run_count": 0,
#         "run_results": [],
#         "max_runs": max_runs,
#     }
#     graph = build_graph()
#     thread_config = {"configurable": {"thread_id": "knowmat2_workflow"}}
#     # Execute the graph.  We ignore the yielded intermediate values because
#     # this pipeline does not stream results in real time.
#     for _ in graph.stream(state, thread_config, stream_mode="values"):
#         pass
#     final_state = graph.get_state(thread_config).values
#     final_data = final_state.get("final_data", {})
#     rationale = final_state.get("rationale", "")
#     flag = final_state.get("flag", False)
#     # Persist the final JSON result to disk
#     out_dir = settings.output_dir
#     os.makedirs(out_dir, exist_ok=True)
#     # Name the output file based on the input PDF
#     base_name = os.path.splitext(os.path.basename(pdf_path))[0]
#     output_path = os.path.join(out_dir, f"{base_name}_extraction.json")
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(final_data, f, ensure_ascii=False, indent=2)
#     print(f"✅ Saved extracted data to {output_path}")
#     # Write rationale separately
#     rationale_path = os.path.join(out_dir, f"{base_name}_rationale.txt")
#     with open(rationale_path, "w", encoding="utf-8") as f:
#         f.write(rationale)
#     print(f"ℹ️  Rationale written to {rationale_path}")
#     # Also write a summary of run_results for debugging
#     runs_path = os.path.join(out_dir, f"{base_name}_runs.json")
#     with open(runs_path, "w", encoding="utf-8") as f:
#         json.dump(final_state.get("run_results", []), f, ensure_ascii=False, indent=2)
#     return {
#         "final_data": final_data,
#         "rationale": rationale,
#         "flag": flag,
#     }

def run(
    pdf_path: str,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    max_runs: int = 3,
    subfield_model: Optional[str] = None,
    extraction_model: Optional[str] = None,
    evaluation_model: Optional[str] = None,
    manager_model: Optional[str] = None,
    flagging_model: Optional[str] = None,
) -> dict:
    """Run the full KnowMat 2.0 pipeline on a given PDF and write results.
    
    Parameters
    ----------
    pdf_path : str
        Path to the materials science paper in PDF format.
    output_dir : Optional[str]
        Directory where results will be saved. If not provided, uses default from settings.
    model_name : Optional[str]
        Override the base model (e.g., "gpt-4", "gpt-5-mini"). If not provided, uses default from settings.
    max_runs : int
        Maximum number of extraction/evaluation cycles. Default is 3.
    subfield_model : Optional[str]
        Model for subfield detection agent. Overrides default.
    extraction_model : Optional[str]
        Model for extraction agent. Overrides default.
    evaluation_model : Optional[str]
        Model for evaluation agent. Overrides default.
    manager_model : Optional[str]
        Model for manager aggregation agent. Overrides default.
    flagging_model : Optional[str]
        Model for flagging/quality assessment agent. Overrides default.
    
    Returns
    -------
    dict
        Results dictionary containing final_data, flag, output_dir, etc.
    """
    
    # Load environment variables if a .env was found
    if _env_path:
        print(f"Loaded environment variables from: {_env_path}")

    # Apply any CLI overrides to settings
    overrides = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if model_name:
        overrides["model_name"] = model_name
    
    # Per-agent model overrides
    if subfield_model:
        overrides["subfield_model"] = subfield_model
    if extraction_model:
        overrides["extraction_model"] = extraction_model
    if evaluation_model:
        overrides["evaluation_model"] = evaluation_model
    if manager_model:
        overrides["manager_model"] = manager_model
    if flagging_model:
        overrides["flagging_model"] = flagging_model
    
    if overrides:
        new_settings = Settings(**overrides)
        settings.__dict__.update(new_settings.model_dump())
    
    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"   Subfield Detection: {settings.subfield_model}")
    print(f"   Extraction:         {settings.extraction_model}")
    print(f"   Evaluation:         {settings.evaluation_model}")
    print(f"   Aggregation:        rule-based (no LLM)")
    print(f"   Validation:         {settings.manager_model}")
    print(f"   Flagging:           {settings.flagging_model}")

    # Create paper-specific subfolder BEFORE running the graph
    # This ensures all nodes (including docling parser) save to the correct location
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_output_dir = os.path.join(settings.output_dir, base_name)
    os.makedirs(paper_output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {paper_output_dir}\n")

    # Prepare the initial state with paper-specific output directory
    state: KnowMatState = {
        "pdf_path": pdf_path,
        "output_dir": paper_output_dir,  # Use paper-specific directory
        "run_count": 0,
        "run_results": [],
        "max_runs": max_runs,
    }

    graph = build_graph()
    # Use a unique thread ID for each paper to avoid context accumulation
    thread_id = f"knowmat2_{base_name}_{uuid.uuid4().hex[:8]}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Execute the graph
    for _ in graph.stream(state, thread_config, stream_mode="values"):
        pass

    final_state = graph.get_state(thread_config).values
    final_data = final_state.get("final_data", {})
    flag = final_state.get("flag", False)

    # Apply property standardization using PostProcessor
    if final_data and final_data.get("compositions"):
        print(f"\nStandardizing property names...")
        try:
            # Initialize PostProcessor with properties.json
            properties_file = os.path.join(os.path.dirname(__file__), "properties.json")
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Skipping property standardization.")
            elif not os.path.exists(properties_file):
                print(f"Warning: properties.json not found at {properties_file}. Skipping.")
            else:
                processor = PostProcessor(
                    properties_file=properties_file,
                    api_key=api_key,
                    gpt_model="gpt-5-mini"
                )
                
                # Create a mock structure for the PostProcessor
                # It expects: [{"data": {"compositions": [...]}}]
                mock_result = [{"data": final_data}]
                
                # Update with standardized properties
                processor.update_extracted_json(mock_result)
                
                # Extract the updated final_data back (it's modified in place)
                final_data = mock_result[0]["data"]
                
                # Print statistics
                processor._print_match_stats()
                print(f"Property standardization complete\n")
        except Exception as e:
            print(f"Warning: Property standardization failed: {str(e)}")
            print(f"Continuing with non-standardized properties.\n")

    # Write final extraction JSON
    output_path = os.path.join(paper_output_dir, f"{base_name}_extraction.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"Saved extraction to {output_path}")

    # Write comprehensive analysis report
    report_path = os.path.join(paper_output_dir, f"{base_name}_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        _write_comprehensive_report(f, final_state)
    print(f"Saved analysis report to {report_path}")

    # Write run details for debugging
    runs_path = os.path.join(paper_output_dir, f"{base_name}_runs.json")
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(final_state.get("run_results", []), f, ensure_ascii=False, indent=2)

    return {
        "final_data": final_data,
        "flag": flag,
        "output_dir": paper_output_dir,  # Return paper-specific directory
        "final_confidence_score": final_state.get("final_confidence_score"),
        "aggregation_rationale": final_state.get("aggregation_rationale"),
        "human_review_guide": final_state.get("human_review_guide"),
    }


def _write_comprehensive_report(f, final_state):
    """Write a comprehensive analysis report to the file handle."""
    
    f.write(f"{'='*80}\n")
    f.write(f"KNOWMAT 2.0 EXTRACTION ANALYSIS REPORT\n")
    f.write(f"{'='*80}\n\n")
    
    # Final Assessment Section
    f.write(f"{'█'*80}\n")
    f.write(f"FINAL ASSESSMENT\n")
    f.write(f"{'█'*80}\n\n")
    
    final_confidence = final_state.get("final_confidence_score", "N/A")
    needs_review = final_state.get("needs_human_review", True)
    confidence_rationale = final_state.get("confidence_rationale", "")
    
    f.write(f"Final Confidence Score: {final_confidence}\n")
    f.write(f"Human Review Required: {'Yes' if needs_review else 'No'}\n")
    f.write(f"Review Flag: {'🚩 FLAGGED' if needs_review else '✅ PASSED'}\n\n")
    
    if confidence_rationale:
        f.write(f"Confidence Assessment Rationale:\n")
        f.write(f"{'-'*40}\n")
        wrapped_rationale = textwrap.fill(confidence_rationale, width=80, subsequent_indent="  ")
        f.write(f"{wrapped_rationale}\n\n")
    
    # Manager Aggregation Section
    f.write(f"{'█'*80}\n")
    f.write(f"MANAGER AGGREGATION ANALYSIS\n")
    f.write(f"{'█'*80}\n\n")
    
    aggregation_rationale = final_state.get("aggregation_rationale", "")
    if aggregation_rationale:
        f.write(f"How Data Was Combined:\n")
        f.write(f"{'-'*25}\n")
        wrapped_aggregation = textwrap.fill(aggregation_rationale, width=80, subsequent_indent="  ")
        f.write(f"{wrapped_aggregation}\n\n")
    
    # Human Review Guide Section
    f.write(f"{'█'*80}\n")
    f.write(f"HUMAN REVIEW GUIDANCE\n")
    f.write(f"{'█'*80}\n\n")
    
    human_review_guide = final_state.get("human_review_guide", "")
    if human_review_guide:
        f.write(f"Items to Double-Check:\n")
        f.write(f"{'-'*22}\n")
        wrapped_guide = textwrap.fill(human_review_guide, width=80, subsequent_indent="  ")
        f.write(f"{wrapped_guide}\n\n")
    
    # Per-Run Analysis Section
    f.write(f"{'█'*80}\n")
    f.write(f"INDIVIDUAL RUN ANALYSIS\n")
    f.write(f"{'█'*80}\n\n")
    
    run_results = final_state.get("run_results", [])
    for i, run in enumerate(run_results, 1):
        f.write(f"{'▓'*60}\n")
        f.write(f"RUN {run.get('run_id', i)} DETAILS\n")
        f.write(f"{'▓'*60}\n")
        f.write(f"Confidence Score: {run.get('confidence_score', 0.0):.2f}\n\n")
        
        rationale_text = run.get('rationale', 'N/A')
        f.write(f"Evaluation Rationale:\n")
        wrapped_rationale = textwrap.fill(rationale_text, width=80, subsequent_indent="  ")
        f.write(f"  {wrapped_rationale}\n\n")
        
        missing = run.get('missing_fields') or []
        if missing:
            f.write(f"Missing Fields ({len(missing)} items):\n")
            for j, field in enumerate(missing[:15], 1):  # Show first 15
                f.write(f"  {j:2d}. {field}\n")
            if len(missing) > 15:
                f.write(f"      ... and {len(missing) - 15} more items\n")
            f.write("\n")
        
        hallucinated = run.get('hallucinated_fields') or []
        if hallucinated:
            f.write(f"Hallucinated Fields ({len(hallucinated)} items):\n")
            for j, field in enumerate(hallucinated[:15], 1):  # Show first 15
                f.write(f"  {j:2d}. {field}\n")
            if len(hallucinated) > 15:
                f.write(f"      ... and {len(hallucinated) - 15} more items\n")
            f.write("\n")
        
        suggestions = run.get('suggested_prompt')
        if suggestions and suggestions.strip():
            f.write(f"Improvement Suggestions:\n")
            wrapped_suggestions = textwrap.fill(suggestions, width=80, subsequent_indent="  ")
            f.write(f"  {wrapped_suggestions}\n\n")
        
        # Add composition summary
        extracted_data = run.get('extracted_data', {})
        compositions = extracted_data.get('compositions', [])
        f.write(f"Compositions Extracted: {len(compositions)}\n")
        if compositions:
            f.write(f"Sample Compositions: ")
            comp_names = [comp.get('composition', 'Unknown') for comp in compositions[:3]]
            f.write(f"{', '.join(comp_names)}")
            if len(compositions) > 3:
                f.write(f" (and {len(compositions) - 3} more)")
            f.write("\n")
        f.write("\n")
    
    # Summary Statistics
    f.write(f"{'█'*80}\n")
    f.write(f"EXTRACTION STATISTICS\n")
    f.write(f"{'█'*80}\n\n")
    
    if run_results:
        scores = [run.get('confidence_score', 0.0) for run in run_results]
        f.write(f"Number of Extraction Runs: {len(run_results)}\n")
        f.write(f"Average Run Confidence: {sum(scores)/len(scores):.2f}\n")
        f.write(f"Best Run Confidence: {max(scores):.2f}\n")
        f.write(f"Worst Run Confidence: {min(scores):.2f}\n\n")
        
        total_missing = sum(len(run.get('missing_fields') or []) for run in run_results)
        total_hallucinated = sum(len(run.get('hallucinated_fields') or []) for run in run_results)
        f.write(f"Total Missing Fields Across Runs: {total_missing}\n")
        f.write(f"Total Hallucinated Fields Across Runs: {total_hallucinated}\n\n")
    
    final_data = final_state.get("final_data", {})
    final_compositions = final_data.get('compositions', [])
    f.write(f"Final Extracted Compositions: {len(final_compositions)}\n")
    if final_compositions:
        total_props = sum(len(comp.get('properties_of_composition', [])) for comp in final_compositions)
        f.write(f"Total Properties in Final Result: {total_props}\n")
    
    f.write(f"\n{'='*80}\n")
    f.write(f"END OF REPORT\n")
    f.write(f"{'='*80}\n")