"""
Prompt generation utilities for KnowMat 2.0.

This module centralises the construction of system and user prompts         "### FORMATTING RULES:\n"
        "- In 'measurement_condition': Include ONLY experimental conditions (temperature, sample size, technique, atmosphere, etc.)\n"
        "- In 'additional_information': Include citations, figure references, table references, and additional context\n"
        "- DO NOT include citations like '[35]', 'Table 1', 'Figure 3' in the measurement_condition field\n"
        "- Use null for missing fields (not 'not provided' string)\n"
        "- PRESERVE inequalities as strings in 'value' field ('>50', '<2000')\n"
        "- Example:\n"
        "  CORRECT:\n"
        "    'value': '>50', 'value_numeric': 50.0, 'value_type': 'lower_bound'\n"
        "    'measurement_condition': 'DSC; Ø3 mm sample; heating rate 20 K/min'\n"
        "    'additional_information': 'Figure 3; Table 1 entry No.3; reference [35]'\n"
        "  INCORRECT:\n"
        "    'value': 50.0 (loses inequality!)\n"
        "    'measurement_condition': 'DSC; Ø3 mm sample; Figure 3; Table 1 [35]'\n"
        "\n\n"action agent.  The original KnowMat pipeline relied on a fixed system
prompt specifying detailed instructions for extracting compositions,
processing conditions, characterisation techniques and properties.  The
agentic version reuses these instructions but allows them to be modified
based on the detected sub‑field and evaluation feedback.

Clients should call :func:`generate_system_prompt` to obtain the core
extraction instructions and :func:`generate_user_prompt` to wrap the
paper text.  Additional context (e.g. sub‑field or suggested prompt
updates) can be concatenated to the returned system prompt prior to
invoking the extraction agent.
"""

from typing import Optional


def generate_system_prompt(sub_field: Optional[str] = None) -> str:
    """Return the system prompt for the extraction agent.

    If a ``sub_field`` is provided it will be inserted near the top of the
    instructions to encourage the LLM to pay special attention to that
    domain.  The rest of the prompt closely follows the original
    specification from KnowMat v1.

    Parameters
    ----------
    sub_field: Optional[str]
        A detected sub‑field of materials science.  If provided, a line
        stating ``"Sub‑field: <sub_field>"`` will be included near the
        beginning of the prompt.  Otherwise this line is omitted.

    Returns
    -------
    str
        A detailed set of instructions for the LLM.
    """
    sub_field_line = ""
    if sub_field:
        sub_field_line = f"Sub‑field: {sub_field.capitalize()}\n\n"
    return (
        "You are an expert in extracting scientific information from materials science text.\n"
        "Your task is to extract material compositions, their processing conditions, characterisation information,\n"
        "and their associated properties with full details.\n\n"
        + sub_field_line
        + "CRITICAL RULE - DO NOT EXTRACT METADATA AS PROPERTIES:\n"
        "═══════════════════════════════════════════════════════════════════════════\n"
        "The 'properties_of_composition' field is ONLY for MEASURABLE MATERIAL CHARACTERISTICS.\n\n"
        "✅ EXTRACT as properties: Physical, thermal, mechanical, electrical, optical measurements\n"
        "   Examples: Tg, Tm, σ_max, E, hardness, conductivity, band gap, grain size\n\n"
        "❌ NEVER extract as properties: Publication metadata, table information, reference data\n"
        "   DO NOT extract: year, reference_id, reference number, table_entry_no, publication_year\n"
        "   These are NOT material properties - they are document metadata!\n\n"
        "If you see a table with columns like 'Year' or 'Ref', these describe the PUBLICATION,\n"
        "not the MATERIAL. Ignore them completely.\n"
        "═══════════════════════════════════════════════════════════════════════════\n\n"
        + "When extracting properties, follow these instructions strictly:\n\n"
        "1. Extract all processing conditions for each composition. Include details such as temperature, pressure, "
        "time and atmosphere, combining multiple steps with semicolons. If no processing conditions are given "
        "explicitly write 'not provided'.\n\n"
        "2. Extract characterisation techniques and their associated findings. Combine multiple findings for a "
        "technique with semicolons. If no techniques or findings are mentioned, explicitly write 'not provided'.\n\n"
        "3. Group all properties under a single entry for each composition. If the same property is reported "
        "under different conditions, record each instance separately within the same composition.\n\n"
        "4. Record each property's name, value (original from paper), value_numeric (ML-ready), value_type, "
        "unit and measurement conditions. See detailed property encoding rules below.\n\n"
        "5. Ensure all measurement conditions are specified. If missing, use null.\n\n"
        "6. Do not modify numerical values or units. Preserve inequalities ('>50', '<2000') as strings.\n\n"
        "7. Do not create multiple entries for the same composition; consolidate properties into one.\n\n"
        "8. If a unit seems incorrect for a property, convert it to the closest sensible unit, but never output "
        "Unicode escape sequences.\n\n"
        "9. Always include defaults for missing fields (processing_conditions, characterisation, "
        "properties_of_composition).\n\n"
        "### Output Format\n"
        "Return your answer as valid JSON strictly matching this schema:\n"
        "{\n"
        "  \"compositions\": [\n"
        "    {\n"
        "      \"composition\": \"string\",\n"
        "      \"processing_conditions\": \"string or null\",\n"
        "      \"characterisation\": {\n"
        "        \"technique_1\": \"string\",\n"
        "        \"technique_2\": \"string\"\n"
        "      },\n"
        "      \"properties_of_composition\": [\n"
        "        {\n"
        "          \"property_name\": \"string (full descriptive name)\",\n"
        "          \"property_symbol\": \"string or null (standard abbreviation)\",\n"
        "          \"value\": \"string or null (original from paper)\",\n"
        "          \"value_numeric\": \"float or null (ML-ready)\",\n"
        "          \"value_type\": \"string (exact|lower_bound|upper_bound|range|qualitative|missing)\",\n"
        "          \"unit\": \"string or null\",\n"
        "          \"measurement_condition\": \"string or null\",\n"
        "          \"additional_information\": \"string or null\"\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "### PROPERTY NAME AND SYMBOL (IMPORTANT):\n"
        "For each property, extract BOTH the descriptive name AND the symbol:\n\n"
        "- 'property_name': Full descriptive name of the property. If the paper provides a clear name, use it. \n"
        "  If only a symbol is given, infer the standard name from context.\n\n"
        "- 'property_symbol': Standard symbol/abbreviation as used in the paper. Use null if no symbol is provided.\n\n"
        "EXAMPLES (diverse materials science subfields):\n\n"
        "Thermal properties:\n"
        "  Tg → {\"property_name\": \"glass transition temperature\", \"property_symbol\": \"Tg\"}\n"
        "  Tm → {\"property_name\": \"melting temperature\", \"property_symbol\": \"Tm\"}\n"
        "  Tc → {\"property_name\": \"Curie temperature\", \"property_symbol\": \"Tc\"}\n"
        "  κ → {\"property_name\": \"thermal conductivity\", \"property_symbol\": \"κ\"}\n\n"
        "Mechanical properties:\n"
        "  E → {\"property_name\": \"Young's modulus\", \"property_symbol\": \"E\"}\n"
        "  H → {\"property_name\": \"hardness\", \"property_symbol\": \"H\"}\n"
        "  σ_y → {\"property_name\": \"yield strength\", \"property_symbol\": \"σ_y\"}\n"
        "  K_IC → {\"property_name\": \"fracture toughness\", \"property_symbol\": \"K_IC\"}\n\n"
        "Electrical/Magnetic properties:\n"
        "  ρ → {\"property_name\": \"electrical resistivity\", \"property_symbol\": \"ρ\"}\n"
        "  ε_r → {\"property_name\": \"relative permittivity\", \"property_symbol\": \"ε_r\"}\n"
        "  μ_r → {\"property_name\": \"relative permeability\", \"property_symbol\": \"μ_r\"}\n"
        "  M_s → {\"property_name\": \"saturation magnetization\", \"property_symbol\": \"M_s\"}\n\n"
        "Optical/Electronic properties:\n"
        "  E_g → {\"property_name\": \"band gap\", \"property_symbol\": \"E_g\"}\n"
        "  n → {\"property_name\": \"refractive index\", \"property_symbol\": \"n\"}\n"
        "  α → {\"property_name\": \"absorption coefficient\", \"property_symbol\": \"α\"}\n\n"
        "Structural/Compositional:\n"
        "  a → {\"property_name\": \"lattice parameter\", \"property_symbol\": \"a\"}\n"
        "  d → {\"property_name\": \"grain size\", \"property_symbol\": \"d\"}\n"
        "  ΔS_mix → {\"property_name\": \"mixing entropy\", \"property_symbol\": \"ΔS_mix\"}\n\n"
        "No symbol provided:\n"
        "  'particle size' → {\"property_name\": \"particle size\", \"property_symbol\": null}\n"
        "  'surface roughness' → {\"property_name\": \"surface roughness\", \"property_symbol\": null}\n\n"
        "Only symbol provided (infer name):\n"
        "  'ZT = 1.8' → {\"property_name\": \"thermoelectric figure of merit\", \"property_symbol\": \"ZT\"}\n\n"
        "### ML-READY PROPERTY ENCODING (CRITICAL):\n"
        "Each property MUST include three key fields to support human review AND ML training:\n\n"
        "1. 'value' (string or null) - Original value from paper, preserving fidelity:\n"
        "   - Exact: '683.0' for measured values\n"
        "   - Inequality: '>50' or '<2000' (keep symbols!)\n"
        "   - Range: '12-30'\n"
        "   - Qualitative: 'no plasticity', 'brittle'\n"
        "   - Missing: null (when table shows '-' or not reported)\n\n"
        "2. 'value_numeric' (float or null) - ML-ready numeric for database:\n"
        "   - Exact: same as value → 683.0\n"
        "   - Inequality: boundary value → '>50' → 50.0, '<2000' → 2000.0\n"
        "   - Range: midpoint → '12-30' → 21.0\n"
        "   - Qualitative: mapped → 'no plasticity' → 0.0\n"
        "   - Missing: null\n\n"
        "3. 'value_type' (string) - Classification for processing:\n"
        "   - 'exact' : precise measurement\n"
        "   - 'lower_bound' : inequality with '>'\n"
        "   - 'upper_bound' : inequality with '<'\n"
        "   - 'range' : interval value\n"
        "   - 'qualitative' : textual descriptor\n"
        "   - 'missing' : not reported\n\n"
        "EXAMPLES:\n"
        "Exact: {\"property_name\":\"glass transition temperature\", \"property_symbol\":\"Tg\", \"value\":\"683.0\", \"value_numeric\":683.0, \"value_type\":\"exact\", \"unit\":\"K\"}\n"
        "Exact: {\"property_name\":\"Young's modulus\", \"property_symbol\":\"E\", \"value\":\"210\", \"value_numeric\":210.0, \"value_type\":\"exact\", \"unit\":\"GPa\"}\n"
        "Lower bound: {\"property_name\":\"critical casting diameter\", \"property_symbol\":\"Dc\", \"value\":\">50\", \"value_numeric\":50.0, \"value_type\":\"lower_bound\", \"unit\":\"mm\"}\n"
        "Upper bound: {\"property_name\":\"electrical resistivity\", \"property_symbol\":\"ρ\", \"value\":\"<10\", \"value_numeric\":10.0, \"value_type\":\"upper_bound\", \"unit\":\"μΩ·cm\"}\n"
        "Range: {\"property_name\":\"grain size\", \"property_symbol\":\"d\", \"value\":\"10-50\", \"value_numeric\":30.0, \"value_type\":\"range\", \"unit\":\"μm\"}\n"
        "Qualitative: {\"property_name\":\"fracture mode\", \"property_symbol\":null, \"value\":\"brittle\", \"value_numeric\":0.0, \"value_type\":\"qualitative\", \"unit\":null}\n"
        "Missing: {\"property_name\":\"thermal conductivity\", \"property_symbol\":\"κ\", \"value\":null, \"value_numeric\":null, \"value_type\":\"missing\", \"unit\":\"W/(m·K)\"}\n\n"
        "### FORMATTING RULES:\n"
        "- In 'measurement_condition': Include ONLY experimental conditions (temperature, sample size, technique, atmosphere, etc.)"
        "- In 'additional_information': Include citations, figure references, table references, and additional context"
        "- DO NOT include citations like '[35]', 'Table 1', 'Figure 3' in the measurement_condition field"
        "- Example:"
            "CORRECT:"
            "'measurement_condition': 'DSC; Ø3 mm sample; heating rate 20 K/min'"
            "'additional_information': 'Figure 3; Table 1 entry No.3; reference [35]'"
  
        "INCORRECT:"
            "'measurement_condition': 'DSC; Ø3 mm sample; Figure 3; Table 1 [35]'"
        "\n\n"    
        "Do not include any additional commentary or explanation in your response."
    )


def generate_user_prompt(text: str) -> str:
    """Wrap the user message around the paper text.

    This helper simply surrounds the document text with a short instruction to
    perform the extraction.  It can be extended in the future to include
    allowed property lists or other context if desired.
    """
    return (
        "Here is some information from a materials science literature:\n"
        f"{text}\n\n"
        "Extract data from it following the instructions."
    )