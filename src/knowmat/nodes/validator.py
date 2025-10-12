"""
Validation Agent (Stage 2 of Two-Stage Manager)

This agent's responsibility is to validate and correct the merged data from Stage 1.
It contains ALL the hallucination correction logic, safety checks, and quality validation
from the original Manager agent.

Responsibilities:
- Detect hallucinations using evaluation feedback
- Correct hallucinations when evaluation provides the fix
- Validate ML-ready format compliance
- Generate human review guide
- Apply ALL safety checks (placeholder detection, lazy fallback, etc.)
- Retry mechanism with stronger prompts if needed

Inputs from Stage 1:
- aggregated_data: Merged compositions from all runs
- aggregation_notes: How the data was merged
- run_results: Original evaluation feedback (for hallucination detection)
"""

import json
from typing import Dict, Any

from knowmat.extractors import manager_extractor, ManagerFeedback, CompositionList
from knowmat.states import KnowMatState


def validate_and_correct(state: KnowMatState) -> Dict[str, Any]:
    """Validate merged data and correct hallucinations.
    
    Takes the aggregated data from Stage 1 and:
    1. Checks for hallucinations using evaluation feedback
    2. Corrects hallucinations when possible (evaluation tells us how)
    3. Validates ML-ready format
    4. Generates human review guide
    5. Applies all safety checks from original manager
    
    Parameters
    ----------
    state : KnowMatState
        Current workflow state containing:
        - aggregated_data: Merged data from Stage 1
        - aggregation_notes: Merge strategy notes
        - run_results: Evaluation feedback
        - paper_text: Original paper content
    
    Returns
    -------
    dict
        Updates containing:
        - final_data: Validated and corrected compositions
        - aggregation_rationale: Full explanation (merge + validation)
        - human_review_guide: Specific items to verify
    """
    aggregated_data = state.get("aggregated_data", {})
    aggregation_notes = state.get("aggregation_notes", "")
    run_results = state.get("run_results", [])
    paper_text = state.get("paper_text", "")
    
    if not aggregated_data or not aggregated_data.get("compositions"):
        # Empty aggregation - fallback
        print("Warning: Stage 1 aggregation returned empty data. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    print(f"\nValidation Stage 2:")
    print(f"  Validating {len(aggregated_data.get('compositions', []))} aggregated compositions...")
    
    # Build validation prompt with ALL the hallucination correction logic
    # This is the COMPLETE prompt from the original manager
    validation_prompt = _build_validation_prompt(
        aggregated_data,
        aggregation_notes,
        run_results,
        paper_text
    )
    
    # Invoke the validation LLM (first attempt)
    result = manager_extractor.invoke(validation_prompt)
    response = result.get("responses", [None])[0]
    
    if response is None:
        print("Warning: Validation LLM returned no response. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    # Convert response to dict
    if isinstance(response, ManagerFeedback):
        validation_dict = response.model_dump()
    else:
        validation_dict = dict(response)
    
    # Extract results
    final_extracted_data = validation_dict.get("final_extracted_data", {})
    aggregation_rationale = validation_dict.get("aggregation_rationale", "")
    human_review_guide = validation_dict.get("human_review_guide", "")
    
    # Ensure proper data structure
    if isinstance(final_extracted_data, dict) and 'compositions' in final_extracted_data:
        final_data = final_extracted_data
    elif hasattr(final_extracted_data, 'compositions'):
        final_data = {"compositions": final_extracted_data.compositions}
    else:
        print("Warning: Validation returned invalid data structure. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAFETY CHECKS (ALL from original manager - PRESERVED)
    # ═══════════════════════════════════════════════════════════════════════
    compositions = final_data.get("compositions", [])
    avg_run_confidence = sum(r.get("confidence_score", 0.0) for r in run_results) / max(len(run_results), 1)
    
    # Detect failure patterns - RELAXED VERSION (original was too aggressive)
    # Only flag as placeholder if there's CLEAR evidence of failure
    has_no_compositions = not compositions or len(compositions) == 0
    has_trivial_rationale = len(aggregation_rationale.strip()) < 100  # Raised from 50
    has_todo_markers = any(marker in aggregation_rationale for marker in ["TODO", "[INSERT", "PLACEHOLDER_", "XXX"])
    has_trivial_review = human_review_guide.strip() in ["1) Verify.", "Verify.", ""]
    
    is_placeholder_response = (
        has_no_compositions or
        has_trivial_rationale or
        has_todo_markers or
        has_trivial_review
    )
    # NOTE: Removed "placeholder" keyword check - too many false positives
    #       (validator might mention "placeholder values" in rationale legitimately)
    
    # Detect lazy fallback (EXACT logic from original manager)
    is_lazy_fallback = (
        "Fallback: Selected run" in aggregation_rationale and
        len(compositions) > 0 and
        avg_run_confidence > 0.85 and
        len(run_results) > 1
    )
    
    if is_placeholder_response:
        print("  Warning: Validator returned empty/placeholder response.")
        print(f"    Compositions: {len(compositions)}, Rationale length: {len(aggregation_rationale)}")
        print(f"    Triggers: no_comps={has_no_compositions}, trivial_rationale={has_trivial_rationale}, ")
        print(f"              todo_markers={has_todo_markers}, trivial_review={has_trivial_review}")
        if has_todo_markers:
            print(f"    First 200 chars of rationale: {aggregation_rationale[:200]}")
        print("  Using fallback aggregation.")
        return _fallback_to_best_run(run_results)
    
    if is_lazy_fallback:
        print("  Warning: Validator chose lazy fallback despite good data.")
        print(f"    Avg run confidence: {avg_run_confidence:.2f}")
        print("  Retrying with stronger instructions...")
        
        # Retry with stronger prompt (EXACT logic from original manager)
        retry_result = _retry_validation_with_explicit_schema(
            aggregated_data,
            aggregation_notes,
            run_results,
            paper_text,
            validation_prompt
        )
        
        if retry_result:
            print("  Retry successful!")
            return retry_result
        else:
            print("  Retry failed. Using fallback.")
            return _fallback_to_best_run(run_results)
    
    # Success - validation completed
    print(f"  Validation complete: {len(compositions)} compositions validated")
    
    # Combine aggregation notes with validation rationale
    combined_rationale = (
        f"STAGE 1 - AGGREGATION:\n{aggregation_notes}\n\n"
        f"STAGE 2 - VALIDATION & CORRECTION:\n{aggregation_rationale}"
    )
    
    return {
        "final_data": final_data,
        "aggregation_rationale": combined_rationale,
        "human_review_guide": human_review_guide,
    }


def _build_validation_prompt(aggregated_data, aggregation_notes, run_results, paper_text):
    """Build the complete validation prompt with ALL hallucination correction logic.
    
    This contains the ENTIRE prompt from the original manager, focused on
    validation and correction rather than aggregation.
    """
    prompt = (
        "You are a materials science data validation specialist.\n\n"
        
        "YOUR ROLE:\n"
        "You receive merged extraction data that has already been aggregated from multiple runs.\n"
        "Your job is to VALIDATE this data and CORRECT any errors based on evaluation feedback.\n\n"
        
        "YOUR RESPONSIBILITIES:\n"
        "1) Review the aggregated data for hallucinations (using evaluation feedback)\n"
        "2) CORRECT hallucinations when the feedback tells you how to fix them\n"
        "3) Ensure all properties follow ML-ready format (value/value_numeric/value_type)\n"
        "4) Generate specific human review guidance for remaining uncertainties\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "HALLUCINATION CORRECTION & TRACKING LOGIC (FOLLOW EXACTLY):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "For EACH field/value in the aggregated data, follow this decision tree:\n\n"
        
        "STEP 1: Check hallucination status across runs\n"
        "   - If NO run marks it as hallucinated → KEEP IT\n"
        "   - If ANY run marks it as hallucinated → Go to STEP 2\n\n"
        
        "STEP 2: Read the hallucination description for correction clues\n"
        "   The evaluation rationale explains WHY and often suggests the fix:\n"
        "   - 'Converted >50 mm to 50.0 mm' → Correct value is '>50 mm' (string)\n"
        "   - 'Converted <2000 MPa to 2000.0 MPa' → Correct value is '<2000 MPa' (string)\n"
        "   - 'Used placeholder 0.0 for missing value' → No correct value available\n"
        "   - 'Invented numeric for boolean XRD confirmation' → Should be boolean/text, not numeric\n\n"
        
        "STEP 3: Determine if you can CORRECT the hallucination\n"
        "   A) Hallucination description tells you the correct value:\n"
        "      → CORRECT IT in final output\n"
        "      → Document correction in aggregation_rationale\n"
        "      → Add verification note in human_review_guide\n\n"
        
        "   B) Hallucination was corrected in a later run:\n"
        "      → Use the CORRECTED value from that run\n"
        "      → Example: Run 1 has '50.0 mm' (hallucinated), Run 2 has '>50 mm' (not hallucinated) → Use Run 2\n\n"
        
        "   C) Cannot determine the correct value:\n"
        "      → Follow standard hallucination tracking (STEP 4)\n\n"
        
        "STEP 4: Standard hallucination tracking\n"
        "   - Hallucinated in some runs but NOT others → KEEP from non-hallucinated runs\n"
        "   - Hallucinated in ALL runs → EXCLUDE (truly unreliable)\n\n"
        
        "STEP 5: Missing fields\n"
        "   - If missing in one run but present in another:\n"
        "     * Check if the present version is hallucinated\n"
        "     * If NOT hallucinated → KEEP it (fills a gap)\n"
        "     * If hallucinated → Try to CORRECT it (go to STEP 3)\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "CORRECTION EXAMPLES (from diverse materials science domains):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "Example 1: Inequality correction - Ceramics (CAN CORRECT)\n"
        "  Aggregated data: grain_size = 200.0 nm\n"
        "  Hallucination note: 'Converted >200 nm to exact value 200.0'\n"
        "  → CORRECT to '>200 nm' (string with inequality preserved)\n\n"
        
        "Example 2: Inequality correction - Metals (CAN CORRECT)\n"
        "  Aggregated data: yield_strength = 500.0 MPa\n"
        "  Hallucination note: 'Converted <500 MPa to exact value 500.0'\n"
        "  → CORRECT to '<500 MPa' (string with inequality preserved)\n\n"
        
        "Example 3: Placeholder for missing value - Polymers (CANNOT CORRECT)\n"
        "  Aggregated data: glass_transition_temperature = 0.0 K\n"
        "  Hallucination note: 'Placeholder 0.0 used for missing Tg'\n"
        "  → Check other runs; if all have placeholders, CORRECT to null (value_type='missing')\n\n"
        
        "Example 4: Qualitative misencoding - Semiconductors (CAN CORRECT)\n"
        "  Aggregated data: band_gap_type = 1.0 eV\n"
        "  Hallucination note: 'Invented numeric 1.0 for direct/indirect band gap type'\n"
        "  → CORRECT: Remove numeric property, keep textual value 'direct' or 'indirect'\n\n"
        
        "Example 5: Range correction - Biomaterials (CAN CORRECT)\n"
        "  Aggregated data: pore_size = 150.0 μm\n"
        "  Hallucination note: 'Converted range 100-200 μm to midpoint 150.0'\n"
        "  → CORRECT: value='100-200', value_numeric=150.0, value_type='range'\n\n"
        
        "Example 6: Corrected in later run - Composites (USE CORRECTED VERSION)\n"
        "  Run 1: fiber_diameter = 7.0 μm (hallucinated: 'Converted 5-9 μm to 7.0')\n"
        "  Run 2: fiber_diameter = '5-9 μm' (not hallucinated)\n"
        "  → Use Run 2's version: value='5-9', value_numeric=7.0, value_type='range'\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "ML-READY PROPERTY FIELD REQUIREMENTS (CRITICAL FOR DATABASE):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "Each property requires THREE fields for human review AND ML training:\n\n"
        
        "1. 'value' (string or null) - HIGH-FIDELITY original value from paper:\n"
        "   - Exact measurements: '683.0'\n"
        "   - Inequalities: '>50' or '<2000' (preserve symbols!)\n"
        "   - Ranges: '12-30'\n"
        "   - Qualitative: 'no plasticity', 'brittle', 'amorphous'\n"
        "   - Missing: null (when table shows '-' or not reported)\n\n"
        
        "2. 'value_numeric' (float or null) - ML-ready numeric:\n"
        "   Exact: 683.0 | Inequalities: 50.0, 2000.0 (boundary) | Ranges: 21.0 (midpoint)\n"
        "   Qualitative: 0.0 | Missing: null\n\n"
        
        "3. 'value_type' (string) - CLASSIFICATION for downstream processing:\n"
        "   - 'exact' : precise measurement (e.g., Tg = 683.0 K)\n"
        "   - 'lower_bound' : inequality with '>' (e.g., Dc > 50 mm)\n"
        "   - 'upper_bound' : inequality with '<' (e.g., σ_max < 2000 MPa)\n"
        "   - 'range' : interval value (e.g., Dc = 12-30 mm)\n"
        "   - 'qualitative' : textual descriptor (e.g., 'no plasticity')\n"
        "   - 'missing' : not reported in paper (value and value_numeric are null)\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "ML-READY PROPERTY ENCODING EXAMPLES (diverse materials domains):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "Example 1: Exact measurement - Superconductors\n"
        "  Property: critical_temperature\n"
        "  Paper value: '92.5 K'\n"
        "  Encoding:\n"
        '    value: "92.5"\n'
        "    value_numeric: 92.5\n"
        '    value_type: "exact"\n'
        "    unit: 'K'\n\n"
        
        "Example 2: Lower bound - Magnetic materials\n"
        "  Property: coercivity\n"
        "  Paper value: '>1000 Oe'\n"
        "  Encoding:\n"
        '    value: ">1000"\n'
        "    value_numeric: 1000.0\n"
        '    value_type: "lower_bound"\n'
        "    unit: 'Oe'\n\n"
        
        "Example 3: Upper bound - Optical materials\n"
        "  Property: absorption_edge\n"
        "  Paper value: '<400 nm'\n"
        "  Encoding:\n"
        '    value: "<400"\n'
        "    value_numeric: 400.0\n"
        '    value_type: "upper_bound"\n'
        "    unit: 'nm'\n\n"
        
        "Example 4: Range - Battery materials\n"
        "  Property: capacity_retention\n"
        "  Paper value: '85-92%'\n"
        "  Encoding:\n"
        '    value: "85-92"\n'
        "    value_numeric: 88.5\n"
        '    value_type: "range"\n'
        "    unit: '%'\n\n"
        
        "Example 5: Qualitative - Catalysts\n"
        "  Property: selectivity\n"
        "  Paper value: 'highly selective for CO2 reduction'\n"
        "  Encoding:\n"
        '    value: "highly selective for CO2 reduction"\n'
        "    value_numeric: 0.0\n"
        '    value_type: "qualitative"\n'
        "    unit: null\n\n"
        
        "Example 6: Missing - Thin films\n"
        "  Property: deposition_rate\n"
        "  Paper value: Not reported\n"
        "  Encoding:\n"
        "    value: null\n"
        "    value_numeric: null\n"
        '    value_type: "missing"\n'
        "    unit: null\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "YOUR VALIDATION TASK:\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "INPUTS YOU RECEIVE:\n"
        "1) Aggregated data - Already merged compositions from all extraction runs\n"
        "2) Aggregation notes - How the data was merged in Stage 1\n"
        "3) Evaluation feedback - Hallucination reports from each run\n\n"
        
        "WHAT YOU MUST DO:\n"
        "1. ✅ Review each property in the aggregated data\n"
        "2. ✅ Check if any run flagged it as hallucinated (in evaluation feedback)\n"
        "3. ✅ If hallucinated AND correction is clear → CORRECT it\n"
        "4. ✅ Validate ML-ready format (value/value_numeric/value_type triplets)\n"
        "5. ✅ Preserve inequalities as strings ('>200', '<500')\n"
        "6. ✅ Generate specific review items for humans to verify\n\n"
        
        "WHAT YOUR OUTPUT SHOULD CONTAIN:\n"
        "- final_extracted_data: The validated and corrected compositions (CompositionList format)\n"
        "- aggregation_rationale: Detailed explanation of what you validated/corrected\n"
        "- human_review_guide: Numbered list of specific items needing verification\n\n"
        
        "CRITICAL: Your final_extracted_data MUST follow the CompositionList schema exactly:\n"
        "{\n"
        '  "compositions": [\n'
        "    {\n"
        '      "composition": "Al0.5CoCrFeNi",\n'
        '      "processing_conditions": "Arc melting + casting",\n'
        '      "characterisation": {"XRD": "FCC structure", "SEM": "Dendritic"},\n'
        '      "properties_of_composition": [\n'
        "        {\n"
        '          "property_name": "hardness",\n'
        '          "property_symbol": "HV",\n'
        '          "value": "450",\n'
        "          \"value_numeric\": 450.0,\n"
        '          "value_type": "exact",\n'
        '          "unit": "HV"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n\n"
    )
    
    # Add aggregation notes from Stage 1
    prompt += f"STAGE 1 AGGREGATION NOTES:\n{aggregation_notes}\n\n"
    
    # Add the aggregated data
    prompt += f"AGGREGATED DATA TO VALIDATE:\n"
    prompt += f"{'─' * 80}\n"
    prompt += f"{json.dumps(aggregated_data, ensure_ascii=False, indent=2)}\n"
    prompt += f"{'─' * 80}\n\n"
    
    # Add run evaluation feedback for hallucination detection
    prompt += "EVALUATION FEEDBACK (for hallucination correction):\n\n"
    for i, run in enumerate(run_results, 1):
        prompt += f"{'═' * 80}\n"
        prompt += f"RUN {run.get('run_id', i)} FEEDBACK\n"
        prompt += f"{'═' * 80}\n"
        prompt += f"Confidence: {run.get('confidence_score', 0.0):.2f}\n\n"
        
        prompt += f"Rationale:\n{run.get('rationale', 'No rationale')}\n\n"
        
        missing = run.get('missing_fields', [])
        if missing:
            prompt += f"Missing Fields ({len(missing)}):\n"
            for field in missing[:15]:
                prompt += f"  - {field}\n"
            if len(missing) > 15:
                prompt += f"  ... and {len(missing) - 15} more\n"
            prompt += "\n"
        
        hallucinated = run.get('hallucinated_fields', [])
        if hallucinated:
            prompt += f"HALLUCINATED FIELDS ({len(hallucinated)}) - READ FOR CORRECTION CLUES:\n"
            for j, field in enumerate(hallucinated[:15], 1):
                prompt += f"  {j:2d}. {field}\n"
            if len(hallucinated) > 15:
                prompt += f"       ... and {len(hallucinated) - 15} more\n"
            prompt += "\n"
        else:
            prompt += "No hallucinated fields in this run\n\n"
    
    prompt += (
        f"{'═' * 80}\n"
        "BEGIN VALIDATION:\n"
        f"{'═' * 80}\n\n"
        "Review the aggregated data against the evaluation feedback below.\n"
        "Apply hallucination corrections where the feedback indicates the fix.\n"
        "Ensure all properties have proper ML-ready encoding.\n"
        "Return the validated data with any necessary corrections.\n\n"
        "In your aggregation_rationale, explain:\n"
        "- What corrections you applied (with justification from evaluation feedback)\n"
        "- What data you verified as correct\n"
        "- Any remaining uncertainties or edge cases\n\n"
        "In your human_review_guide, provide:\n"
        "- Numbered list of specific items to verify\n"
        "- Focus on corrected hallucinations, unusual values, or ambiguous data\n"
        "- Be specific: mention composition names, property names, and values\n\n"
        "Use the tool to provide your validated output.\n"
        f"{'═' * 80}\n"
    )
    
    return prompt


def _retry_validation_with_explicit_schema(aggregated_data, aggregation_notes, run_results, paper_text, original_prompt):
    """Retry validation with stronger instructions if lazy fallback detected.
    
    This is the EXACT retry logic from the original manager.
    """
    retry_prompt = original_prompt + (
        "\n\n"
        f"{'═' * 80}\n"
        "RETRY - VALIDATION REQUIRED\n"
        f"{'═' * 80}\n\n"
        "Your previous response was not satisfactory. You returned a lazy fallback\n"
        "instead of validating the aggregated data.\n\n"
        
        "WHAT YOU NEED TO DO:\n"
        "1. Read the AGGREGATED DATA structure shown above\n"
        "2. Review each property against the EVALUATION FEEDBACK\n"
        "3. Apply corrections for any hallucinations\n"
        "4. Return a COMPLETE validated dataset\n\n"
        
        "The data has already been merged. Your job is VALIDATION only.\n"
        "Provide a thorough validation with specific corrections and rationale.\n"
        f"{'═' * 80}\n"
    )
    
    # Invoke validation LLM with retry prompt
    result = manager_extractor.invoke(retry_prompt)
    response = result.get("responses", [None])[0]
    
    if response is None:
        return None
    
    # Convert response
    if isinstance(response, ManagerFeedback):
        validation_dict = response.model_dump()
    else:
        validation_dict = dict(response)
    
    # Extract results
    final_extracted_data = validation_dict.get("final_extracted_data", {})
    aggregation_rationale = validation_dict.get("aggregation_rationale", "")
    human_review_guide = validation_dict.get("human_review_guide", "")
    
    # Ensure proper structure
    if isinstance(final_extracted_data, dict) and 'compositions' in final_extracted_data:
        final_data = final_extracted_data
    elif hasattr(final_extracted_data, 'compositions'):
        final_data = {"compositions": final_extracted_data.compositions}
    else:
        return None
    
    compositions = final_data.get("compositions", [])
    
    # Check if retry still failed
    if "Fallback: Selected run" in aggregation_rationale:
        return None
    
    if not compositions or len(aggregation_rationale.strip()) < 50:
        return None
    
    # Success - combine with aggregation notes
    combined_rationale = (
        f"STAGE 1 - AGGREGATION:\n{aggregation_notes}\n\n"
        f"STAGE 2 - VALIDATION & CORRECTION (RETRY):\n{aggregation_rationale}"
    )
    
    return {
        "final_data": final_data,
        "aggregation_rationale": combined_rationale,
        "human_review_guide": human_review_guide,
    }


def _fallback_to_best_run(run_results):
    """Fallback aggregation if validation fails completely.
    
    This is the EXACT fallback logic from the original manager.
    """
    if not run_results:
        return {
            "final_data": {"compositions": []},
            "aggregation_rationale": "Validation failed with no run data available.",
            "human_review_guide": "Manual review required - validation pipeline failed.",
        }
    
    sorted_runs = sorted(run_results, key=lambda r: r.get("confidence_score", 0.0), reverse=True)
    best_run = sorted_runs[0]
    
    final_data = best_run.get("extracted_data", {})
    
    return {
        "final_data": final_data,
        "aggregation_rationale": f"Fallback: Validation failed. Selected run {best_run.get('run_id')} with highest confidence {best_run.get('confidence_score', 0.0):.2f}.",
        "human_review_guide": "Review extraction quality due to validation fallback.",
    }
