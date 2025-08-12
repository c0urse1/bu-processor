#!/usr/bin/env python3
"""Test that the patch fix works by checking imports directly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pinecone_manager_import():
    """Test that PineconeManager can be imported from enhanced_integrated_pipeline."""
    try:
        # Import the module directly without going through bu_processor init
        sys.path.append(str(project_root / "bu_processor" / "pipeline"))
        
        # Import the enhanced_integrated_pipeline module 
        import enhanced_integrated_pipeline as eip
        
        # Check if PineconeManager is available for patching
        if hasattr(eip, 'PineconeManager'):
            print("✅ PineconeManager can be imported from enhanced_integrated_pipeline")
            return True
        else:
            print("❌ PineconeManager NOT available in enhanced_integrated_pipeline")
            return False
            
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chatbot_integration_import():
    """Test that ChatbotIntegration can be imported from enhanced_integrated_pipeline."""
    try:
        # Import the module directly without going through bu_processor init
        sys.path.append(str(project_root / "bu_processor" / "pipeline"))
        
        # Import the enhanced_integrated_pipeline module 
        import enhanced_integrated_pipeline as eip
        
        # Check if ChatbotIntegration is available for patching
        if hasattr(eip, 'ChatbotIntegration'):
            print("✅ ChatbotIntegration can be imported from enhanced_integrated_pipeline")
            return True
        else:
            print("❌ ChatbotIntegration NOT available in enhanced_integrated_pipeline")
            return False
            
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_patch_targets():
    """Test that test patch targets are available."""
    try:
        # Import the module directly
        sys.path.append(str(project_root / "bu_processor" / "pipeline"))
        import enhanced_integrated_pipeline as eip
        
        patch_targets = [
            ('PineconeManager', 'bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager'),
            ('ChatbotIntegration', 'bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration')
        ]
        
        all_available = True
        
        for class_name, patch_path in patch_targets:
            if hasattr(eip, class_name):
                print(f"✅ {patch_path} - Available for patching")
            else:
                print(f"❌ {patch_path} - NOT available for patching")
                all_available = False
        
        return all_available
        
    except Exception as e:
        print(f"❌ Error testing patch targets: {e}")
        return False

def main():
    """Main test function."""
    print("="*70)
    print("🔧 TESTING PINECONE/PIPELINE PATCHING FIXES")
    print("="*70)
    
    print("\n1. Testing PineconeManager import:")
    test1 = test_pinecone_manager_import()
    
    print("\n2. Testing ChatbotIntegration import:")  
    test2 = test_chatbot_integration_import()
    
    print("\n3. Testing all patch targets:")
    test3 = test_patch_targets()
    
    print(f"\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if test1 and test2 and test3:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Tests can now patch bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager")
        print("✅ Tests can now patch bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration")
        print("\nThe AttributeError should be fixed!")
        return True
    else:
        print("❌ Some tests failed. The patch fix needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
