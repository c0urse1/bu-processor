#!/usr/bin/env python3
"""
BU-Processor Demo Runner
=======================

Simple demo runner to showcase system capabilities.
"""

def run_demo():
    """Run the enhanced semantic deduplication demo"""
    try:
        from bu_processor.pipeline.simhash_semantic_deduplication import demo_enhanced_semantic_simhash
        print("🚀 BU-Processor Demo Starting...")
        demo_enhanced_semantic_simhash()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -e .")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    run_demo()
