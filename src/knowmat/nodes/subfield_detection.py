"""
Node for detecting the materials science sub‑field and updating the prompt.

This node calls the ``subfield_extractor`` defined in
:mod:`knowmat2.extractors` to analyse the paper text and determine the
appropriate niche domain.  It returns both the detected sub‑field and an
updated extraction prompt tailored to that sub‑field.  The updated prompt
should be concatenated with any existing prompt modifications stored in
``state['updated_prompt']``.
"""

from knowmat.extractors import subfield_extractor, SubFieldDetection
from knowmat.states import KnowMatState


def detect_sub_field(state: KnowMatState) -> dict:
    """Detect the materials science sub‑field and return prompt updates.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state, must contain ``paper_text``.

    Returns
    -------
    dict
        Updates containing ``sub_field`` and ``updated_prompt``.  The
        returned ``updated_prompt`` includes any prior prompt stored on
        ``state['updated_prompt']`` concatenated with the new suggestion.
    """
    paper_text = state.get("paper_text", "")
    # Compose a prompt instructing the LLM to detect the sub‑field and update the prompt
    prompt = (
        "You are a materials science domain expert. Given the following paper text, "
        "determine which niche sub‑field the work belongs to. The valid choices are: "
        "experimental, computational, simulation, machine learning, or hybrid. "
        "Return your answer using the tool with two fields: (1) sub_field and (2) updated_prompt. "
        "The updated_prompt should start with a short instruction that emphasises the selected sub_field "
        "and explains how it influences the extraction (e.g. suggesting to pay more attention to modelling details "
        "for computational papers, simulation parameters for simulation papers or describing the learning algorithm "
        "for machine learning papers). Do not include any extraneous information.\n\n"
        "Paper text:\n"
        f"{paper_text}"
    )
    # Invoke the extractor
    result = subfield_extractor.invoke(prompt)
    # TrustCall returns a dict with a 'responses' key containing the parsed objects
    response = result.get("responses", [None])[0]
    if not response:
        # In the unlikely event no response is returned, fall back to a sensible default
        sub_field = "experimental"
        updated = state.get("updated_prompt", "")
    else:
        # response is a SubFieldDetection instance
        # For compatibility, convert to dict if necessary
        if isinstance(response, SubFieldDetection):
            sub_field = response.sub_field
            updated = response.updated_prompt
        else:
            # Assume dict
            sub_field = response.get("sub_field")
            updated = response.get("updated_prompt", "")
    # Prepend any existing prompt updates
    prior_update = state.get("updated_prompt", "")
    if prior_update:
        new_prompt = prior_update.strip() + "\n\n" + updated.strip()
    else:
        new_prompt = updated.strip()
    return {"sub_field": sub_field, "updated_prompt": new_prompt}