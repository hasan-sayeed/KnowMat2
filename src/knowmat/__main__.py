"""
Entry point for running the KnowMat 2.0 pipeline via the command line.

Usage
-----
::

    python -m knowmat --pdf-folder path/to/pdfs [--output-dir out] [--max-runs 3]

This will parse all PDFs in the given folder, run the agentic extraction workflow
and write the results to the specified output directory. Each PDF is processed
sequentially. The final JSON, rationale and intermediate run records are saved
separately for each paper.
"""

import argparse
import os
from pathlib import Path

from knowmat.orchestrator import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the KnowMat 2.0 extraction pipeline with advanced PDF parsing.")
    parser.add_argument("--pdf-folder", required=True, help="Path to the folder containing PDF files to process.")
    parser.add_argument("--output-dir", default=None, help="Directory to write outputs to (default: ./knowmat_output).")
    parser.add_argument("--max-runs", type=int, default=3, help="Maximum number of extraction/evaluation cycles.")
    
    # Per-agent model overrides
    parser.add_argument("--subfield-model", default=None, help="Model for subfield detection agent (default: gpt-5-mini).")
    parser.add_argument("--extraction-model", default=None, help="Model for extraction agent (default: gpt-5).")
    parser.add_argument("--evaluation-model", default=None, help="Model for evaluation agent (default: gpt-5).")
    parser.add_argument("--manager-model", default=None, help="Model for validation agent (Stage 2: hallucination correction) (default: gpt-5).")
    parser.add_argument("--flagging-model", default=None, help="Model for flagging/quality assessment agent (default: gpt-5-mini).")
    
    args = parser.parse_args(argv)
    
    # Get all PDF files from the folder
    pdf_folder = Path(args.pdf_folder)
    if not pdf_folder.exists():
        print(f"Error: PDF folder not found: {pdf_folder}")
        return
    
    if not pdf_folder.is_dir():
        print(f"Error: Path is not a directory: {pdf_folder}")
        return
    
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in: {pdf_folder}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    print("=" * 60)
    
    results_summary = []
    
    # Process each PDF sequentially
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
        print(f"{'='*60}\n")
        
        try:
            result = run(
                pdf_path=str(pdf_path),
                output_dir=args.output_dir,
                model_name=None,  # Use defaults from settings
                max_runs=args.max_runs,
                subfield_model=args.subfield_model,
                extraction_model=args.extraction_model,
                evaluation_model=args.evaluation_model,
                manager_model=args.manager_model,
                flagging_model=args.flagging_model,
            )
            
            # Print a short summary to stdout
            flag_str = "[FLAGGED]" if result.get("flag") else "[OK]"
            compositions_count = len(result.get('final_data', {}).get('compositions', []))
            
            print(f"\nFinished extraction: {pdf_path.name}")
            print(f"   Status: {flag_str}")
            print(f"   Output: {result.get('output_dir')}")
            print(f"   Compositions: {compositions_count}")
            
            results_summary.append({
                'pdf': pdf_path.name,
                'success': True,
                'flag': result.get('flag'),
                'compositions': compositions_count,
                'output_dir': result.get('output_dir')
            })
            
        except Exception as e:
            print(f"\nError processing {pdf_path.name}: {str(e)}")
            results_summary.append({
                'pdf': pdf_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}\n")
    
    successful = sum(1 for r in results_summary if r['success'])
    failed = len(results_summary) - successful
    
    print(f"Total PDFs: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        flagged = sum(1 for r in results_summary if r['success'] and r.get('flag'))
        print(f"Flagged for review: {flagged}")
        total_compositions = sum(r.get('compositions', 0) for r in results_summary if r['success'])
        print(f"Total compositions: {total_compositions}")
    
    print(f"\n{'='*60}\n")
    
    # Print individual results
    for r in results_summary:
        if r['success']:
            flag_icon = "[FLAGGED]" if r['flag'] else "[OK]"
            print(f"{flag_icon} {r['pdf']}: {r['compositions']} compositions")
        else:
            print(f"[ERROR] {r['pdf']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":  # pragma: no cover
    main()
