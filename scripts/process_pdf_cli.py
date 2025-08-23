#!/usr/bin/env python3
"""
📄 UNIFIED PDF PROCESSING CLI
============================

Command-line interface using the same process_pdf function as the API.
Supports both interactive and batch processing modes.

Usage:
    python scripts/process_pdf_cli.py file.pdf
    python scripts/process_pdf_cli.py --batch data/pdfs/
    python scripts/process_pdf_cli.py file.pdf --output results/ --no-pinecone
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from bu_processor.ingest import process_pdf
    import structlog
    logger = structlog.get_logger("pdf_cli")
except ImportError as e:
    print(f"❌ Failed to import BU-Processor modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def process_single_pdf(file_path: str, args: argparse.Namespace) -> bool:
    """Process a single PDF file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if not file_path.suffix.lower() == '.pdf':
        print(f"❌ Not a PDF file: {file_path}")
        return False
    
    try:
        print(f"🔄 Processing: {file_path.name}")
        
        result = process_pdf(
            file_path=str(file_path),
            output_dir=args.output if args.output else None,
            store_in_pinecone=args.pinecone,
            store_in_sqlite=args.sqlite
        )
        
        # Print results
        print(f"✅ Success: {file_path.name}")
        print(f"   📊 Classification: {result['classification']['predicted_label']} ({result['classification']['confidence']:.2%})")
        print(f"   📝 Text length: {result['text_length']:,} chars")
        
        # Storage results
        if 'sqlite' in result['storage']:
            sqlite_result = result['storage']['sqlite']
            if sqlite_result['status'] == 'success':
                print(f"   💾 SQLite: Document ID {sqlite_result['document_id']}")
            else:
                print(f"   ❌ SQLite: {sqlite_result['error']}")
        
        if 'pinecone' in result['storage']:
            pinecone_result = result['storage']['pinecone']
            if pinecone_result['status'] == 'success':
                print(f"   🌲 Pinecone: {pinecone_result['chunks_stored']} chunks stored")
            else:
                print(f"   ❌ Pinecone: {pinecone_result['error']}")
        
        if args.output:
            print(f"   📁 Files saved to: {args.output}")
        
        if args.verbose:
            # Print detailed results
            print(f"\n📋 Detailed Results:")
            print(json.dumps(result, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process {file_path.name}: {e}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        return False

def process_batch(directory: str, args: argparse.Namespace) -> None:
    """Process all PDF files in a directory"""
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all PDF files
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {directory}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files in {directory}")
    print("=" * 50)
    
    # Process each file
    success_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        success = process_single_pdf(pdf_file, args)
        if success:
            success_count += 1
        else:
            failed_count += 1
        print()  # Empty line between files
    
    # Summary
    print("=" * 50)
    print(f"📊 Batch Processing Summary:")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📄 Total: {len(pdf_files)}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Process PDF files using unified BU-Processor pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_pdf_cli.py document.pdf
  python scripts/process_pdf_cli.py --batch data/pdfs/
  python scripts/process_pdf_cli.py file.pdf --output results/ --no-pinecone
  python scripts/process_pdf_cli.py file.pdf --sqlite-only --verbose
        """
    )
    
    # Input arguments
    parser.add_argument('input', nargs='?', help='PDF file or directory to process')
    parser.add_argument('--batch', action='store_true', help='Process all PDFs in the input directory')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output directory for extracted content and metadata')
    
    # Storage options
    parser.add_argument('--no-pinecone', dest='pinecone', action='store_false', default=True,
                       help='Disable Pinecone vector storage')
    parser.add_argument('--no-sqlite', dest='sqlite', action='store_false', default=True,
                       help='Disable SQLite document storage')
    parser.add_argument('--pinecone-only', action='store_true',
                       help='Store only in Pinecone (disable SQLite)')
    parser.add_argument('--sqlite-only', action='store_true', 
                       help='Store only in SQLite (disable Pinecone)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with detailed results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input:
        parser.print_help()
        print("\n❌ Error: Please specify a PDF file or directory to process")
        sys.exit(1)
    
    # Handle exclusive storage options
    if args.pinecone_only:
        args.pinecone = True
        args.sqlite = False
    elif args.sqlite_only:
        args.pinecone = False
        args.sqlite = True
    
    if not args.pinecone and not args.sqlite:
        print("❌ Error: At least one storage option (Pinecone or SQLite) must be enabled")
        sys.exit(1)
    
    # Show configuration
    print("📄 BU-Processor PDF CLI")
    print("=" * 30)
    print(f"Input: {args.input}")
    print(f"Mode: {'Batch' if args.batch else 'Single file'}")
    print(f"Pinecone: {'✅' if args.pinecone else '❌'}")
    print(f"SQLite: {'✅' if args.sqlite else '❌'}")
    if args.output:
        print(f"Output: {args.output}")
    print("=" * 30)
    print()
    
    # Process input
    try:
        if args.batch:
            process_batch(args.input, args)
        else:
            process_single_pdf(args.input, args)
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
