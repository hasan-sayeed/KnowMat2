"""
LLM-based node for assessing final extraction quality and determining review needs.

The flagging agent evaluates the manager's aggregation result, considers the
complexity of issues, quality of corrections made, and assigns a confidence 
score with review recommendations.
"""

from typing import Dict, Any

from knowmat.extractors import flagging_extractor, FlaggingFeedback
from knowmat.states import KnowMatState


def assess_final_quality(state: KnowMatState) -> Dict[str, Any]:
    """Use LLM to assess the final aggregated extraction quality.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state containing aggregation results and run data.

    Returns
    -------
    dict
        Updates containing ``final_confidence_score``, ``confidence_rationale``, 
        ``needs_human_review``, and ``flag``.
    """
    run_results = state.get("run_results", [])
    aggregation_rationale = state.get("aggregation_rationale", "")
    human_review_guide = state.get("human_review_guide", "")
    final_data = state.get("final_data", {})
    
    if not run_results:
        # No runs to assess
        return {
            "final_confidence_score": 0.5,
            "confidence_rationale": "No evaluation runs available for assessment.",
            "needs_human_review": True,
            "flag": True,
        }
    
    # Prepare the enhanced flagging prompt
    flagging_prompt = (
        "You are a quality assessment expert for materials science data extraction. "
        "Your job is to evaluate the final aggregated result, considering both the original "
        "extraction quality AND the manager's intelligent corrections and aggregation decisions.\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "ASSESSMENT PHILOSOPHY:\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "The final confidence score should reflect:\n"
        "1. CORRECTED QUALITY: Not just raw run quality, but quality AFTER manager corrections\n"
        "2. CORRECTION RELIABILITY: How well-justified and traceable the manager's fixes are\n"
        "3. COMPLETENESS: Whether the final result is comprehensive and rich\n"
        "4. RESIDUAL RISK: What uncertainties remain after aggregation\n\n"
        
        "KEY PRINCIPLE: Successful corrections INCREASE confidence!\n"
        "- If Manager corrected hallucinations using evaluation feedback → GOOD\n"
        "- If Manager excluded unfixable hallucinations appropriately → GOOD\n"
        "- If Manager merged complementary data intelligently → GOOD\n"
        "- If Manager preserved all valid data comprehensively → EXCELLENT\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "ASSESSMENT CRITERIA (Prioritized):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "1. MANAGER CORRECTION QUALITY (40% weight)\n"
        "   - Did the manager successfully correct hallucinations?\n"
        "   - Are corrections well-justified and traceable to evaluation feedback?\n"
        "   - Did the manager follow the inequality/placeholder/boolean correction logic?\n"
        "   - Are corrected values properly documented in aggregation rationale?\n\n"
        
        "2. FINAL DATA COMPLETENESS (30% weight)\n"
        "   - Is the final extraction comprehensive and rich?\n"
        "   - Were valid compositions and properties preserved?\n"
        "   - Did the manager avoid being overly conservative?\n"
        "   - Are there sufficient properties per composition?\n\n"
        
        "3. RUN CONSISTENCY & ORIGINAL QUALITY (20% weight)\n"
        "   - Were the original runs reasonably consistent?\n"
        "   - What was the average confidence across runs?\n"
        "   - How many issues needed correction vs how many were correct from the start?\n\n"
        
        "4. RESIDUAL UNCERTAINTIES (10% weight)\n"
        "   - What uncertainties remain after corrections?\n"
        "   - How complex is the human review guide?\n"
        "   - Are there unresolved conflicts or ambiguities?\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "CONFIDENCE SCORING GUIDELINES (0-1):\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "0.9-1.0: EXCELLENT\n"
        "  - High-quality runs (avg > 0.85) with minimal issues\n"
        "  - Manager successfully corrected all or most hallucinations\n"
        "  - Final data is comprehensive (many compositions with rich properties)\n"
        "  - Human review guide lists only minor verification items\n"
        "  - Corrections are well-documented and traceable\n\n"
        
        "0.8-0.9: GOOD\n"
        "  - Decent run quality (avg 0.75-0.85)\n"
        "  - Manager corrected most hallucinations successfully\n"
        "  - Final data is reasonably complete\n"
        "  - Some uncertainties remain but are clearly documented\n"
        "  - Human review needed for verification of corrections\n\n"
        
        "0.7-0.8: FAIR\n"
        "  - Moderate run quality (avg 0.65-0.75)\n"
        "  - Manager made some corrections but challenges remain\n"
        "  - Final data has gaps or conservative exclusions\n"
        "  - Multiple items need human verification\n"
        "  - Correction decisions are reasonable but not all optimal\n\n"
        
        "0.6-0.7: POOR\n"
        "  - Low run quality (avg < 0.65) OR\n"
        "  - Manager struggled to correct hallucinations OR\n"
        "  - Final data is sparse or overly conservative OR\n"
        "  - Many unresolved conflicts and uncertainties\n\n"
        
        "Below 0.6: VERY POOR\n"
        "  - Extraction runs had major issues\n"
        "  - Manager corrections were ineffective or unclear\n"
        "  - Final data is incomplete or unreliable\n"
        "  - Requires significant human intervention\n\n"
        
        "═══════════════════════════════════════════════════════════════════════════\n"
        "HUMAN REVIEW THRESHOLD:\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        
        "Recommend review (needs_human_review = true) if:\n"
        "- Final confidence score < 0.8 OR\n"
        "- Manager made significant corrections that need verification OR\n"
        "- Human review guide indicates complex verification tasks OR\n"
        "- Manager excluded substantial data due to unfixable hallucinations OR\n"
        "- Runs had major inconsistencies that required difficult decisions OR\n"
        "- Final data completeness is below expectations (e.g., <5 properties per composition on average)\n\n"
        
        "NOTE: Even high-confidence results may need review if corrections were extensive!\n\n"
    )
    
    # Add run summary statistics WITH CORRECTION CONTEXT
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "RUN STATISTICS:\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    if run_results:
        scores = [run.get('confidence_score', 0.0) for run in run_results]
        avg_confidence = sum(scores) / len(scores)
        min_confidence = min(scores)
        max_confidence = max(scores)
        
        flagging_prompt += f"Number of Runs: {len(run_results)}\n"
        flagging_prompt += f"Average Run Confidence: {avg_confidence:.2f}\n"
        flagging_prompt += f"Confidence Range: {min_confidence:.2f} - {max_confidence:.2f}\n"
        flagging_prompt += f"Confidence Spread: {max_confidence - min_confidence:.2f} (consistency indicator)\n\n"
        
        # Count issues across runs (handle None values)
        total_missing = sum(len(run.get('missing_fields') or []) for run in run_results)
        total_hallucinated = sum(len(run.get('hallucinated_fields') or []) for run in run_results)
        
        flagging_prompt += "ORIGINAL EXTRACTION ISSUES (Before Manager Corrections):\n"
        flagging_prompt += f"- Total missing fields across runs: {total_missing}\n"
        flagging_prompt += f"- Total hallucinated fields across runs: {total_hallucinated}\n"
        
        # Provide context about hallucination types
        flagging_prompt += "\nHallucination Breakdown by Run:\n"
        for i, run in enumerate(run_results, 1):
            h_fields = run.get('hallucinated_fields') or []  # Handle None
            if h_fields:
                flagging_prompt += f"  Run {i}: {len(h_fields)} hallucinations\n"
                # Show first few for context
                for j, h in enumerate(h_fields[:3], 1):
                    flagging_prompt += f"    {j}. {h[:100]}{'...' if len(h) > 100 else ''}\n"
                if len(h_fields) > 3:
                    flagging_prompt += f"    ... and {len(h_fields) - 3} more\n"
        flagging_prompt += "\n"
    
    # Add manager's assessment WITH CORRECTION ANALYSIS
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "MANAGER'S AGGREGATION & CORRECTION ANALYSIS:\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    flagging_prompt += "Read the manager's rationale carefully to assess:\n"
    flagging_prompt += "1. What corrections were made (e.g., '50.0 mm' → '>50 mm')\n"
    flagging_prompt += "2. How well corrections are justified\n"
    flagging_prompt += "3. What data was excluded and why\n"
    flagging_prompt += "4. How comprehensive the final aggregation is\n\n"
    
    flagging_prompt += f"Manager's Aggregation Rationale:\n"
    flagging_prompt += f"{'-'*40}\n{aggregation_rationale}\n\n"
    
    # Add human review guide ANALYSIS
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "HUMAN REVIEW REQUIREMENTS:\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    flagging_prompt += "Assess the complexity and scope of verification tasks:\n"
    flagging_prompt += "- Simple verifications (check a specific value) → Lower impact on confidence\n"
    flagging_prompt += "- Complex verifications (resolve conflicts, validate corrections) → Higher impact\n"
    flagging_prompt += "- Extensive verification lists → May indicate lower confidence in aggregation\n\n"
    
    flagging_prompt += f"Human Review Guide:\n"
    flagging_prompt += f"{'-'*40}\n{human_review_guide}\n\n"
    
    # Add final data completeness METRICS
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "FINAL RESULT COMPLETENESS:\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    compositions = final_data.get('compositions', [])
    flagging_prompt += f"Number of Compositions: {len(compositions)}\n"
    
    if compositions:
        total_properties = sum(len(comp.get('properties_of_composition', [])) for comp in compositions)
        total_ns_properties = sum(len(comp.get('non_standard_properties_of_composition', [])) for comp in compositions)
        avg_props = total_properties / len(compositions) if compositions else 0
        
        flagging_prompt += f"Total Standard Properties: {total_properties}\n"
        flagging_prompt += f"Total Non-Standard Properties: {total_ns_properties}\n"
        flagging_prompt += f"Average Properties per Composition: {avg_props:.1f}\n\n"
        
        flagging_prompt += "COMPLETENESS INTERPRETATION:\n"
        flagging_prompt += "- >10 props/composition: Excellent completeness\n"
        flagging_prompt += "- 5-10 props/composition: Good completeness\n"
        flagging_prompt += "- 2-5 props/composition: Fair completeness (may be conservative)\n"
        flagging_prompt += "- <2 props/composition: Poor completeness (likely too conservative or issues)\n\n"
        
        # Sample composition for richness check
        if compositions:
            sample = compositions[0]
            flagging_prompt += f"Sample Composition '{sample.get('composition', 'Unknown')}':\n"
            flagging_prompt += f"  Properties: {len(sample.get('properties_of_composition', []))}\n"
            flagging_prompt += f"  Processing: {'Present' if sample.get('processing_conditions') else 'Missing'}\n"
            flagging_prompt += f"  Characterisation: {len(sample.get('characterisation', {}))} techniques\n"
    else:
        flagging_prompt += "⚠️ WARNING: No compositions extracted! This indicates major failure.\n"
    
    flagging_prompt += "\n"
    
    # Final instructions
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "YOUR ASSESSMENT TASK:\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    flagging_prompt += "Based on all the above information:\n\n"
    
    flagging_prompt += "1. EVALUATE MANAGER'S CORRECTION QUALITY\n"
    flagging_prompt += "   - Were hallucinations corrected effectively?\n"
    flagging_prompt += "   - Are corrections well-documented and traceable?\n"
    flagging_prompt += "   - Did manager follow the correction logic appropriately?\n\n"
    
    flagging_prompt += "2. ASSESS FINAL DATA QUALITY\n"
    flagging_prompt += "   - Is the extraction comprehensive (not overly conservative)?\n"
    flagging_prompt += "   - Are there sufficient properties per composition?\n"
    flagging_prompt += "   - Is important data preserved despite hallucinations?\n\n"
    
    flagging_prompt += "3. DETERMINE CONFIDENCE SCORE (0-1)\n"
    flagging_prompt += "   - Weight manager corrections heavily (40%)\n"
    flagging_prompt += "   - Consider final completeness (30%)\n"
    flagging_prompt += "   - Factor in original run quality (20%)\n"
    flagging_prompt += "   - Account for residual uncertainties (10%)\n\n"
    
    flagging_prompt += "4. DECIDE ON HUMAN REVIEW FLAG\n"
    flagging_prompt += "   - Flag if confidence < 0.8\n"
    flagging_prompt += "   - Flag if manager made extensive corrections needing verification\n"
    flagging_prompt += "   - Flag if final completeness is below expectations\n"
    flagging_prompt += "   - Flag if human review guide indicates complex tasks\n\n"
    
    flagging_prompt += "5. WRITE CLEAR RATIONALE\n"
    flagging_prompt += "   - Explain your confidence score reasoning\n"
    flagging_prompt += "   - Reference specific manager decisions (corrections, exclusions)\n"
    flagging_prompt += "   - Note what drove the review flag decision\n"
    flagging_prompt += "   - Be specific about strengths and weaknesses\n\n"
    
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    flagging_prompt += "OUTPUT FORMAT REQUIREMENTS (CRITICAL):\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n\n"
    
    flagging_prompt += "For the 'confidence_rationale' field:\n"
    flagging_prompt += "- Write in paragraph format (flowing text)\n"
    flagging_prompt += "- Focus ONLY on explaining the confidence score\n"
    flagging_prompt += "- Do NOT include suggestions, offers, or additional services\n"
    flagging_prompt += "- Do NOT offer to generate CSV files, tables, or other outputs\n"
    flagging_prompt += "- Keep it concise and professional\n\n"
    
    flagging_prompt += "INCORRECT (DO NOT DO THIS):\n"
    flagging_prompt += '  "...score of 0.82. If you want, I can produce a CSV table..."\n'
    flagging_prompt += '  "...comprehensive result. Let me know if you need further analysis..."\n\n'
    
    flagging_prompt += "CORRECT:\n"
    flagging_prompt += '  "...score of 0.82 based on successful corrections and good completeness."\n\n'
    
    flagging_prompt += "Use the tool to provide your assessment.\n"
    flagging_prompt += "═══════════════════════════════════════════════════════════════════════════\n"
    
    # Invoke the flagging extractor
    result = flagging_extractor.invoke(flagging_prompt)
    response = result.get("responses", [None])[0]
    
    if response is None:
        # Fallback assessment
        avg_confidence = sum(run.get('confidence_score', 0.0) for run in run_results) / len(run_results) if run_results else 0.0
        return {
            "final_confidence_score": avg_confidence,
            "confidence_rationale": "Fallback assessment: averaged run confidence scores.",
            "needs_human_review": avg_confidence < 0.8,
            "flag": avg_confidence < 0.8,
        }
    
    # Convert response to dict
    if isinstance(response, FlaggingFeedback):
        flagging_dict = response.model_dump()
    else:
        flagging_dict = dict(response)
    
    final_confidence = flagging_dict.get("final_confidence_score", 0.0)
    confidence_rationale = flagging_dict.get("confidence_rationale", "")
    needs_review = flagging_dict.get("needs_human_review", False)
    
    return {
        "final_confidence_score": final_confidence,
        "confidence_rationale": confidence_rationale,
        "needs_human_review": needs_review,
        "flag": needs_review,
    }