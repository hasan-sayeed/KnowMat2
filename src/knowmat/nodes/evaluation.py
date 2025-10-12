"""
Node for evaluating an extracted JSON result against the paper text.

This node calls the ``evaluation_extractor`` defined in
:mod:`knowmat2.extractors` to compare the most recent extraction with the
original document.  The extractor should output a confidence score,
a rationale, lists of missing and hallucinated fields, an optional
prompt update and a boolean indicating whether to perform another
extraction.  The evaluation result is appended to ``state['run_results']``
and ``state['run_count']`` is incremented.  ``state['needs_rerun']`` is
set based on the extractor's output.

If the extractor returns no response, the node assumes the extraction
should be accepted as‑is and sets ``needs_rerun`` to False.
"""

import json
from typing import Dict, Any, List

from knowmat.extractors import evaluation_extractor, EvaluationFeedback
from knowmat.states import KnowMatState, EvaluationRun


def evaluate_data(state: KnowMatState) -> Dict[str, Any]:
    """Evaluate the extracted data against the source and decide on a rerun.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state.  Must include ``paper_text`` and
        ``latest_extracted_data``.  ``updated_prompt`` may contain
        previous prompt updates.

    Returns
    -------
    dict
        Updates containing ``run_results``, ``run_count``, ``needs_rerun``
        and possibly an updated ``updated_prompt`` if the evaluation agent
        suggests additional refinements.
    """
    paper_text = state.get("paper_text", "")
    extracted_data = state.get("latest_extracted_data", {})
    run_count = state.get("run_count", 0)
    
    # Compose a prompt instructing the evaluation agent with EXPLICIT FIX FORMAT
    evaluation_prompt = (
        "You are an evaluation agent tasked with assessing the correctness of extracted JSON data "
        "against the original paper. Your goal is to identify missing fields (properties or details "
        "present in the paper but absent from the extraction) and hallucinated fields (items in the "
        "extraction not supported by the text).\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "HALLUCINATION REPORTING FORMAT (CRITICAL):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "For EACH hallucinated field, use this exact format:\n"
        "[FIELD_PATH] | HALLUCINATED: [what was extracted] | PAPER_SAYS: [what paper actually says] | FIX: [how to correct it]\n\n"
        
        "Examples of GOOD hallucination reports:\n\n"
        
        "Example 1: Inequality conversion\n"
        "Vit1.properties.Dc | HALLUCINATED: 50.0 mm (exact numeric) | PAPER_SAYS: '>50 mm' (inequality in Table 1) | FIX: Replace with string '>50 mm'\n\n"
        
        "Example 2: Invented value\n"
        "Ti16.7Zr16.7Nb16.7Cu16.7Ni16.7Be16.7.properties.ΔS_mix | HALLUCINATED: 13.38 J/(mol·K) | PAPER_SAYS: Not listed in Table 1 for this composition | FIX: Remove this property entirely\n\n"
        
        "Example 3: Placeholder\n"
        "Ti32.8Zr30.2Ni5.3Cu9Be22.7.properties.Tm | HALLUCINATED: 0.0 K (placeholder) | PAPER_SAYS: '-' (not reported in Table 1) | FIX: Set to null or remove\n\n"
        
        "Example 4: Wrong conversion\n"
        "Vit1.properties.σ_max | HALLUCINATED: -2000.0 MPa (negative value) | PAPER_SAYS: '<2000 MPa' (inequality in Table 1) | FIX: Replace with string '<2000 MPa'\n\n"
        
        "Example 5: Boolean to numeric\n"
        "Ti20Zr20Cu20Ni20Be20.properties.XRD_amorphous | HALLUCINATED: 1.0 a.u. (invented numeric) | PAPER_SAYS: XRD shows amorphous (qualitative) | FIX: Move to characterisation.XRD as text description 'amorphous'\n\n"
        
        "Example 6: Wrong assignment\n"
        "Ti20Hf20Cu20Ni20Be20.properties.ΔS_mix | HALLUCINATED: 14.9 J/(mol·K) (copied from another composition) | PAPER_SAYS: Not listed for this composition in Table 1 | FIX: Remove or assign to correct composition\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "MISSING FIELD REPORTING FORMAT:\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "For missing fields, describe:\n"
        "1. What field/data is missing\n"
        "2. Where in the paper it appears (Table X, Figure Y, page Z, etc.)\n"
        "3. What value/content should be extracted\n\n"
        
        "IMPORTANT: When describing missing data, distinguish between:\n"
        "- PROPERTIES (measurable material characteristics): Tg, Tm, σ_max, Dc, etc.\n"
        "- METADATA (publication/table info): Year, reference numbers, table entry numbers\n"
        "  → Metadata should NEVER be extracted as 'properties_of_composition'\n"
        "  → Year, reference_id, table_entry_no are NOT material properties!\n\n"
        
        "Example:\n"
        "Vit1 composition from Table 1 | MISSING: Entire composition with properties | PAPER_LOCATION: Table 1, row 1 | SHOULD_EXTRACT: Composition Zr41.2Ti13.8Cu12.5Ni10Be22.5 with properties: Tg=625 K, Tx=705 K, Tm=937 K, Tl=993 K, σ_max='<2000 MPa', Dc='>50 mm', ΔS_mix=13.38 J/(mol·K). Note: Publication year (2013) and reference ([4]) are metadata, not material properties.\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "YOUR TASK:\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "1. Compare the extracted JSON against the paper text carefully\n"
        "2. For hallucinated fields: Use the exact format above with HALLUCINATED, PAPER_SAYS, and FIX\n"
        "3. For missing fields: Describe what's missing, where it is in the paper, and what should be extracted\n"
        "4. Provide a confidence_score (0-1) based on accuracy\n"
        "5. Write a rationale summarizing your evaluation\n"
        "6. Suggest update_prompt text to improve future extractions\n"
        "7. Set needs_rerun=true if significant issues require another extraction\n"
        "8. Check ML-ready field compliance:\n"
        "   - Each property should have 'value', 'value_numeric', and 'value_type'\n"
        "   - 'value' should preserve inequalities as strings ('>50', '<2000')\n"
        "   - 'value_numeric' should extract boundary numbers\n"
        "   - 'value_type' should correctly classify as exact/lower_bound/upper_bound/range/qualitative/missing\n"
        "   - Report violations as hallucinations with FIX instructions\n\n"
        
        "IMPORTANT:\n"
        "- Be SPECIFIC: Instead of 'incorrect value', say 'HALLUCINATED: 50.0 | PAPER_SAYS: >50 | FIX: use string >50'\n"
        "- Include LOCATION: Reference table numbers, figure numbers, or page locations\n"
        "- Provide ACTIONABLE fixes: Tell exactly what should be done to correct each error\n\n"
        
        "Use the tool to respond with your evaluation.\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "PAPER TEXT:\n"
        "═══════════════════════════════════════════════════════════════════════════\n"
        f"{paper_text}\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "EXTRACTED JSON:\n"
        "═══════════════════════════════════════════════════════════════════════════\n"
        f"{json.dumps(extracted_data, ensure_ascii=False, indent=2)}"
    )
    
    result = evaluation_extractor.invoke(evaluation_prompt)
    response = result.get("responses", [None])[0]
    # Prepare the updated run_results list
    run_results: List[EvaluationRun] = list(state.get("run_results", []))
    if response is None:
        # No evaluation returned; accept the extraction as is
        return {
            "run_results": run_results,
            "run_count": run_count + 1,
            "needs_rerun": False,
        }
    # Convert to a dictionary regardless of underlying type
    if isinstance(response, EvaluationFeedback):
        eval_dict = response.model_dump()
    else:
        eval_dict = dict(response)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAFETY CHECK: Detect incomplete/invalid evaluation responses
    # ═══════════════════════════════════════════════════════════════════════════
    # If the LLM fails to properly invoke the tool or returns an incomplete 
    # response, we detect it here and force a rerun rather than accepting 
    # invalid evaluation data (confidence=0.0 with empty rationale).
    confidence = eval_dict.get("confidence_score", 0.0)
    rationale = eval_dict.get("rationale", "")
    needs_rerun_from_llm = eval_dict.get("needs_rerun", False)
    
    # Check for invalid evaluation: zero confidence AND empty rationale
    # This indicates the LLM failed to properly evaluate (technical error)
    if confidence == 0.0 and not rationale.strip():
        print("⚠️  WARNING: Evaluation agent returned incomplete response (confidence=0.0, empty rationale)")
        print(f"   This appears to be a technical error. Forcing rerun {run_count + 1}/{state.get('max_runs', 3)}")
        
        # Override the evaluation to force a rerun
        eval_dict["rationale"] = (
            "⚠️ WARNING: Evaluation agent returned incomplete response (possible LLM tool invocation failure). "
            "Forcing automatic rerun to obtain valid evaluation."
        )
        eval_dict["needs_rerun"] = True
        eval_dict["confidence_score"] = 0.0  # Keep 0.0 to indicate invalid evaluation
        
        # Set the corrected values for later use
        confidence = 0.0
        rationale = eval_dict["rationale"]
        needs_rerun_from_llm = True
    
    # Compose the evaluation run record
    run_id = run_count + 1
    record: EvaluationRun = {
        "run_id": run_id,
        "confidence_score": confidence,
        "rationale": rationale,
        "missing_fields": eval_dict.get("missing_fields"),
        "hallucinated_fields": eval_dict.get("hallucinated_fields"),
        "suggested_prompt": eval_dict.get("update_prompt"),
        "extracted_data": extracted_data,
    }
    run_results.append(record)
    # Update the extraction prompt if suggested
    updated_prompt = state.get("updated_prompt", "")
    update_text = eval_dict.get("update_prompt")
    if update_text:
        if updated_prompt:
            updated_prompt = updated_prompt.strip() + "\n\n" + update_text.strip()
        else:
            updated_prompt = update_text.strip()
    
    # Use the needs_rerun value (potentially corrected by safety check above)
    needs_rerun = bool(needs_rerun_from_llm)
    
    return {
        "run_results": run_results,
        "run_count": run_id,
        "needs_rerun": needs_rerun,
        "updated_prompt": updated_prompt,
    }