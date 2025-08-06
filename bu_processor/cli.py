#!/usr/bin/env python3
"""
BU-Processor CLI Entry Point
===========================

Command-line interface for the BU-Processor system.
"""

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
    """Run interactive demo"""
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import demo_enhanced_semantic_simhash
        console.print("🎬 Starting BU-Processor Demo...")
        demo_enhanced_semantic_simhash()
    except ImportError as e:
        console.print(f"❌ Demo module not found: {e}", style="red")
    except Exception as e:
        console.print(f"❌ Error running demo: {e}", style="red")

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

def main():
    """Main entry point for CLI"""
    app()

if __name__ == "__main__":
    main()
