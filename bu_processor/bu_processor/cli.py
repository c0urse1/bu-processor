#!/usr/bin/env python3
"""
BU-Processor CLI Entry Point
===========================

Command-line interface for the BU-Processor system.
"""

import json
import logging
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Configure logging centrally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="🚀 BU-Processor - Advanced ML Document Classification System")
console = Console()

@app.command()
def version():
    """Show version information"""
    from bu_processor import __version__, __author__
    console.print(f"🚀 BU-Processor v{__version__}")
    console.print(f"👥 By {__author__}")

@app.command()
def config():
    """Show current configuration"""
    try:
        from bu_processor import get_config
        config = get_config()
        
        table = Table(title="🔧 BU-Processor Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Environment", config.environment.value)
        table.add_row("Debug Mode", str(config.debug))
        table.add_row("Log Level", config.log_level.value)
        table.add_row("Version", config.version)
        
        console.print(table)
        
        # Features
        features_table = Table(title="🎯 Enabled Features")
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Status", style="green")
        
        features = [
            ("Vector DB", config.is_feature_enabled("vector_db")),
            ("Chatbot", config.is_feature_enabled("chatbot")),
            ("Cache", config.is_feature_enabled("cache")),
            ("GPU", config.is_feature_enabled("gpu")),
            ("Semantic Clustering", config.is_feature_enabled("semantic_clustering")),
            ("Semantic Deduplication", config.is_feature_enabled("semantic_deduplication"))
        ]
        
        for feature_name, enabled in features:
            status = "✅ Enabled" if enabled else "❌ Disabled"
            features_table.add_row(feature_name, status)
            
        console.print(features_table)
        
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")

@app.command()
def demo():
    """Show guidance for demo (interactive code demo removed)."""
    console.print("ℹ️  The in-code demo was removed to keep the library lean.")
    console.print("🧪 You can create your own demo by importing EnhancedDeduplicationEngine:")
    console.print("   from bu_processor.pipeline.simhash_semantic_deduplication import EnhancedDeduplicationEngine")
    console.print("   engine = EnhancedDeduplicationEngine(); engine.deduplicate_chunks_semantically([...])")
    console.print("📚 See README for an example snippet.")

@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input PDF file to process"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Process a PDF file"""
    if not input_file.exists():
        console.print(f"❌ Input file not found: {input_file}", style="red")
        raise typer.Exit(1)
    
    if output_dir is None:
        output_dir = Path("output")
    
    console.print(f"📄 Processing: {input_file}")
    console.print(f"📂 Output directory: {output_dir}")
    
    if verbose:
        console.print("🔍 Verbose mode enabled")
    
    # TODO: Implement actual processing
    console.print("⚠️  Processing functionality not yet implemented", style="yellow")

@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="API host address"),
    port: int = typer.Option(8000, "--port", "-p", help="API port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (development)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes")
):
    """Start the REST API server"""
    try:
        import uvicorn
        
        console.print(f"🚀 Starting BU-Processor API server")
        console.print(f"📡 Host: {host}")
        console.print(f"🔌 Port: {port}")
        console.print(f"🔄 Auto-reload: {'✅' if reload else '❌'}")
        console.print(f"👥 Workers: {workers}")
        console.print(f"📚 Documentation: http://{host}:{port}/docs")
        console.print(f"🩺 Health Check: http://{host}:{port}/health")
        
        uvicorn.run(
            "bu_processor.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info",
            access_log=True
        )
        
    except ImportError:
        console.print("❌ uvicorn not installed. Install with: pip install uvicorn", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start API server: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def classify(
    input_data: str = typer.Argument(..., help="Text to classify or path to PDF file"),
    strategy: str = typer.Option("simple", "--strategy", "-s", help="Chunking strategy (simple, semantic, hybrid)"),
    confidence: float = typer.Option(0.8, "--confidence", "-c", help="Confidence threshold"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """Classify text or PDF file"""
    try:
        from bu_processor.pipeline.classifier import RealMLClassifier
        
        console.print("🤖 Initializing classifier...")
        classifier = RealMLClassifier()
        
        # Check if input is a file path
        input_path = Path(input_data)
        if input_path.exists() and input_path.suffix.lower() == '.pdf':
            console.print(f"📄 Processing PDF: {input_path}")
            result = classifier.classify_pdf(input_path)
        else:
            console.print(f"📝 Processing text: {input_data[:50]}{'...' if len(input_data) > 50 else ''}")
            result = classifier.classify_text(input_data)
        
        # Convert result to dict if needed
        if hasattr(result, 'dict'):
            result_data = result.dict()
        else:
            result_data = result
        
        if output_format == "json":
            console.print(json.dumps(result_data, indent=2, default=str))
        else:
            # Table format
            table = Table(title="Classification Result")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Category", str(result_data.get('category', 'N/A')))
            table.add_row("Label", result_data.get('category_label', 'N/A'))
            table.add_row("Confidence", f"{result_data.get('confidence', 0):.3f}")
            table.add_row("Confident", "✅" if result_data.get('is_confident', False) else "❌")
            table.add_row("Input Type", result_data.get('input_type', 'N/A'))
            
            if 'processing_time' in result_data:
                table.add_row("Processing Time", f"{result_data['processing_time']:.3f}s")
            
            console.print(table)
        
    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}", style="red")
        console.print("💡 Install with: pip install -r requirements.txt", style="yellow")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Classification failed: {e}", style="red")
        raise typer.Exit(1)

def main():
    """Main entry point for CLI"""
    app()

if __name__ == "__main__":
    main()
