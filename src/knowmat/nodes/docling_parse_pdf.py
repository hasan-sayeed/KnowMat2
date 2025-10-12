"""
Advanced PDF parsing node using Docling with table extraction and conversion.

This node replaces the simple PyMuPDF parsing with a more sophisticated approach:
1. Uses Docling to extract text and identify tables
2. Saves table images as PNG files
3. Converts table images to HTML using GPT vision models
4. Returns clean markdown with embedded HTML tables
"""

import os
import re
import base64
import mimetypes
import pathlib
import time
import operator
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing_extensions import TypedDict, Annotated

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, TableItem

from knowmat.states import KnowMatState
from knowmat.app_config import settings


class TableConvState(TypedDict):
    """State for individual table conversion."""
    match: str
    path: str
    output_dir: Optional[str]


class DocParseState(TypedDict):
    """State for the document parsing subgraph."""
    input_md: str
    output_md: Optional[str]
    tables: List[Dict[str, str]]
    unique_tables: List[Dict[str, str]]
    conversions: Annotated[List[Dict[str, str]], operator.add]
    output_dir: Optional[str]


# Global cache for table conversions to avoid re-processing
_html_cache: Dict[str, str] = {}

# Regex to find table image links in markdown
TABLE_LINK_RE = re.compile(r"\[\[\[Table\s*([^\]]+)\]\]\]\(([^)]+)\)", re.IGNORECASE)

# Prompt for GPT table conversion
TABLE_CONVERSION_PROMPT = (
    "Can you convert the provided table to html? Please make really sure that you preserve "
    "the exact structure of the table (multi-span columns, merged rows, multi-column headers, "
    "stitch split method label like - back together and stuff like that). Also, even if there's "
    "no clear border in the source table, you can put boarder where you think necessary for easier understanding.\n\n"
    "Just generate the html. You do not need to generate any other description or explanation."
)


def _to_data_url(path: pathlib.Path) -> str:
    """Convert image file to data URL for API calls."""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _format_and_save_html(html: str, image_path: pathlib.Path, output_dir: Optional[str] = None) -> str:
    """Format HTML properly and save to file for inspection."""
    try:
        from bs4 import BeautifulSoup
        # Parse and format HTML
        soup = BeautifulSoup(html, 'html.parser')
        formatted_html = soup.prettify()
    except ImportError:
        # Fallback: basic formatting without BeautifulSoup
        formatted_html = html
        # Remove any existing escaping first
        formatted_html = formatted_html.replace('\\n', '\n').replace('\\t', '\t')
        
        # Add proper newlines after opening tags
        formatted_html = formatted_html.replace('><', '>\n<')
        # Add proper newlines after common closing tags
        for tag in ['</tr>', '</thead>', '</tbody>', '</table>', '</th>', '</td>']:
            formatted_html = formatted_html.replace(tag, tag + '\n')
        # Add proper newlines after opening tags
        for tag in ['<thead>', '<tbody>', '<tr>', '<table']:
            if tag in formatted_html:
                formatted_html = formatted_html.replace(tag, '\n' + tag)
        
        # Clean up multiple newlines
        import re
        formatted_html = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_html)
    
    # Save HTML file for inspection if output_dir is provided
    if output_dir:
        html_dir = Path(output_dir) / "table_html"
        html_dir.mkdir(parents=True, exist_ok=True)
        html_filename = f"{image_path.stem}.html"
        html_file_path = html_dir / html_filename
        
        try:
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_html)
            print(f"Saved formatted HTML to: {html_file_path}")
        except Exception as e:
            print(f"Warning: Could not save HTML file {html_file_path}: {e}")
    
    return formatted_html


def _convert_image_to_html(image_path: pathlib.Path, model: str, api_key: str, output_dir: Optional[str] = None) -> str:
    """Convert table image to HTML using OpenAI vision models."""
    # Check cache first
    key = str(image_path.resolve())
    if key in _html_cache:
        return _html_cache[key]
    
    import openai
    client = openai.OpenAI(api_key=api_key)
    data_url = _to_data_url(image_path)

    delay = 1.0
    for attempt in range(5):
        try:
            # Try responses API first (for GPT-5)
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": TABLE_CONVERSION_PROMPT},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
            )
            html = getattr(resp, "output_text", None)
            if not html:
                html = resp.output[0].content[0].text
            html = html.strip()
            
            # Format HTML properly and save to file for inspection
            formatted_html = _format_and_save_html(html, image_path, output_dir)
            _html_cache[key] = formatted_html
            return formatted_html
        except Exception as e:
            # Retry on transient errors
            if any(x in str(e).lower() for x in ("rate", "timeout", "temporar", "overload", "server", "503", "429")):
                time.sleep(delay)
                delay *= 1.5
                continue
            # Fallback to chat completions API (don't set temperature for GPT-5)
            try:
                chat_params = {
                    "model": "gpt-4o",  # Use gpt-4o as fallback
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TABLE_CONVERSION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    }],
                }
                # Only add temperature for non-GPT-5 models
                if "gpt-5" not in model:
                    chat_params["temperature"] = 0.0
                
                chat = client.chat.completions.create(**chat_params)
                html = chat.choices[0].message.content.strip()
                
                # Format HTML properly and save to file for inspection
                formatted_html = _format_and_save_html(html, image_path, output_dir)
                _html_cache[key] = formatted_html
                return formatted_html
            except Exception:
                pass
            raise
    raise RuntimeError(f"Failed to convert {image_path} after retries.")


def _img_to_b64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_pdf_with_docling(pdf_path: str, output_dir: str) -> tuple[str, List[str]]:
    """
    Extract PDF using Docling and save table images.
    
    Returns:
        tuple: (markdown_text, list_of_table_image_paths)
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    table_img_dir = out_dir / "table_images"
    table_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Docling pipeline
    pipe = PdfPipelineOptions()
    pipe.do_table_structure = True
    pipe.table_structure_options.do_cell_matching = True
    pipe.images_scale = 4.0
    pipe.generate_page_images = True
    pipe.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe)}
    )
    
    # Convert PDF
    result = converter.convert(pdf_path)
    doc = result.document
    stem = pdf_path.stem
    
    # Extract table images
    table_png_paths = []
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            n = len(table_png_paths) + 1
            img = item.get_image(doc)
            png_path = table_img_dir / f"{stem}-table-{n}.png"
            img.save(png_path, "PNG")
            
            # Also save base64 version
            b64_path = table_img_dir / f"{stem}-table-{n}.png.b64.txt"
            with open(b64_path, "w", encoding="utf-8") as bf:
                bf.write(_img_to_b64(img))
            
            table_png_paths.append(png_path.as_posix())
    
    # Get markdown and replace tables with image links
    md = doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
    
    # Define table detection patterns
    FIG_TABLE = re.compile(
        r"<figure\b[^>]*>\s*(?:<!--.*?-->\s*)*<table\b.*?</table>.*?</figure>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    BARE_TABLE = re.compile(
        r"<table\b.*?</table>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    PIPE_TABLE_BLOCK = re.compile(
        r"(?m)(?:^\|[^\n]*\n)+(?:^[^\n]*\|[^\n]*\n)+(?:^\|[^\n]*\n)+",
        flags=re.DOTALL,
    )
    
    def find_pipe_tables(md_str):
        """Find contiguous table-like blocks starting with '|' that contain a separator line."""
        blocks = []
        for m in PIPE_TABLE_BLOCK.finditer(md_str):
            block = m.group(0)
            if re.search(r"^\s*\|?\s*[:\-]+", block, re.MULTILINE):
                blocks.append((m.start(), m.end(), block))
        return blocks
    
    # Collect all table matches
    matches = []
    taken = [False] * (len(md) + 1)
    
    def collect(pattern):
        for m in pattern.finditer(md):
            s, e = m.start(), m.end()
            if any(taken[s:e]):
                continue
            matches.append((s, e))
            for i in range(s, e):
                taken[i] = True
    
    collect(FIG_TABLE)
    collect(BARE_TABLE)
    
    # Handle pipe tables
    for s, e, block in find_pipe_tables(md):
        if not any(taken[s:e]):
            matches.append((s, e))
            for i in range(s, e):
                taken[i] = True
    
    # Sort matches by start position
    matches.sort(key=lambda x: x[0])
    
    # Replace table blocks with image links
    num_replaced = min(len(matches), len(table_png_paths))
    parts = []
    cursor = 0
    for i in range(num_replaced):
        start, end = matches[i]
        parts.append(md[cursor:start])
        link = f"[[[Table {i+1}]]]({table_png_paths[i]})"
        parts.append(f"{link}\n\n")
        cursor = end
    
    parts.append(md[cursor:])
    final_md = "".join(parts)
    
    return final_md, table_png_paths


def table_convert_node(state: TableConvState) -> Dict[str, List[Dict[str, str]]]:
    """Convert a single table image to HTML."""
    p = pathlib.Path(os.path.expanduser(state["path"]))
    if not p.exists():
        html = f"<!-- Missing image {state['path']} -->"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            html = f"<!-- No API key available for {state['path']} -->"
        else:
            try:
                # Pass output directory for saving HTML files
                output_dir = state.get("output_dir")
                html = _convert_image_to_html(p, settings.model_name, api_key, output_dir)
            except Exception as e:
                html = f"<!-- Error converting {state['path']}: {str(e)} -->"
    
    return {"conversions": [{"match": state["match"], "html": html}]}


def parse_tables(state: DocParseState) -> DocParseState:
    """Parse markdown to find table image references."""
    md = state["input_md"]
    found: List[Dict[str, str]] = []
    for m in TABLE_LINK_RE.finditer(md):
        full = m.group(0)
        path = m.group(2).strip().strip('"').strip("'")
        found.append({"match": full, "path": path})
    
    # Deduplicate by path for processing efficiency
    seen = set()
    unique: List[Dict[str, str]] = []
    for t in found:
        if t["path"] not in seen:
            seen.add(t["path"])
            unique.append(t)
    
    state["tables"] = found
    state["unique_tables"] = unique
    state["conversions"] = state.get("conversions", [])
    return state


def initiate_conversions(state: DocParseState):
    """Fan out table conversions."""
    if not state["unique_tables"]:
        return "write_output"
    
    output_dir = state.get("output_dir")
    sends = [
        Send("table_convert_node", {**t, "output_dir": output_dir}) 
        for t in state["unique_tables"]
    ]
    return sends


def write_output(state: DocParseState) -> DocParseState:
    """Replace table image links with HTML."""
    conversions = state.get("conversions", [])
    
    # Build mapping from path to HTML
    path_to_html: Dict[str, str] = {}
    match_to_html: Dict[str, str] = {}
    
    for conv in conversions:
        if "match" in conv and "html" in conv:
            # Ensure HTML is properly unescaped and formatted
            html = conv["html"]
            # Remove any escape characters that might have been added
            html = html.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
            # Add proper markdown spacing around HTML blocks
            html = f"\n\n{html}\n\n"
            
            match_to_html[conv["match"]] = html
            # Extract path for additional mapping
            m = TABLE_LINK_RE.search(conv["match"])
            if m:
                path = m.group(2).strip().strip('"').strip("'")
                path_to_html[path] = html
    
    # Replace all occurrences with HTML
    md = state["input_md"]
    for t in state.get("tables", []):
        html = match_to_html.get(t["match"]) or path_to_html.get(t["path"])
        if html:
            md = md.replace(t["match"], html)
    
    state["output_md"] = md
    return state


def build_table_conversion_graph() -> StateGraph:
    """Build the table conversion subgraph."""
    doc = StateGraph(DocParseState)
    doc.add_node("parse_tables", parse_tables)
    doc.add_node("write_output", write_output)
    doc.add_node("table_convert_node", table_convert_node)
    
    doc.add_edge(START, "parse_tables")
    doc.add_conditional_edges("parse_tables", initiate_conversions, ["write_output", "table_convert_node"])
    doc.add_edge("table_convert_node", "write_output")
    doc.add_edge("write_output", END)
    
    return doc.compile()


def parse_pdf_with_docling(state: KnowMatState) -> dict:
    """
    Parse PDF using Docling with advanced table extraction and conversion.
    
    This replaces the simple PyMuPDF parsing with:
    1. Docling-based PDF parsing for better layout understanding
    2. Table extraction and conversion to high-quality images
    3. AI-powered table-to-HTML conversion
    4. Clean markdown output with embedded HTML tables
    
    Parameters
    ----------
    state: KnowMatState
        Must include ``pdf_path`` pointing to a valid PDF file.
    
    Returns
    -------
    dict
        A dictionary containing the ``paper_text`` extracted from the document
        with tables converted to HTML.
    """
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        raise ValueError("No 'pdf_path' provided in state for parse_pdf_with_docling node.")
    
    # Create output directory
    output_dir = state.get("output_dir", ".")
    parse_output_dir = Path(output_dir) / "docling_parse"
    parse_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract PDF with Docling
        markdown_with_table_links, table_paths = _extract_pdf_with_docling(pdf_path, str(parse_output_dir))
        
        # Step 2: Convert table images to HTML if we have an API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and table_paths:
            # Build and run table conversion graph
            table_graph = build_table_conversion_graph()
            
            result = table_graph.invoke({
                "input_md": markdown_with_table_links,
                "output_md": None,
                "tables": [],
                "unique_tables": [],
                "conversions": [],
                "output_dir": str(parse_output_dir),
            })
            
            final_text = result.get("output_md", markdown_with_table_links)
        else:
            # No API key or no tables - use original markdown
            final_text = markdown_with_table_links
        
        # Step 3: Remove references section (similar to original PyMuPDF parser)
        # Find the references section and remove everything after it
        lines = final_text.split('\n')
        references_found = False
        clean_lines = []
        
        for line in lines:
            # Look for references section headers
            if re.match(r'^#+\s*(references?|bibliography|citations?)\s*$', line.strip(), re.IGNORECASE):
                references_found = True
                break
            if not references_found:
                clean_lines.append(line)
        
        cleaned_text = '\n'.join(clean_lines).strip()
        
        # Save the final markdown for inspection
        try:
            pdf_name = Path(pdf_path).stem
            final_md_path = parse_output_dir / f"{pdf_name}_final_output.md"
            with open(final_md_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Saved final markdown output to: {final_md_path}")
        except Exception as e:
            print(f"Warning: Could not save final markdown file: {e}")
        
        return {"paper_text": cleaned_text}
        
    except Exception as e:
        # Fallback: if Docling fails, could fall back to simple text extraction
        raise RuntimeError(f"Failed to parse PDF with Docling: {str(e)}")