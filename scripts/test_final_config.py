#!/usr/bin/env python3
"""
🧪 FINAL CONFIG TEST
==================
Test that the final unified config approach works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_final_config():
    """Test that enable_vector_db works as the primary field."""
    print("🧪 FINAL CONFIG TEST")
    print("=" * 25)
    
    try:
        # Set environment variable
        os.environ["VECTOR_DB_ENABLE"] = "true"
        
        # Import and get config
        from bu_processor.core.config import get_config
        
        print("✅ Config imported successfully")
        
        # Get the config
        config = get_config()
        print("✅ Config loaded successfully")
        
        # Test vector_db access
        vdb = config.vector_db
        print(f"✅ vector_db object: {type(vdb).__name__}")
        
        # Test the primary field
        try:
            field_value = vdb.enable_vector_db
            print(f"✅ vdb.enable_vector_db: {field_value}")
        except AttributeError as e:
            print(f"❌ vdb.enable_vector_db failed: {e}")
            return False
        
        # Test that it loads from environment
        if field_value == True:
            print("✅ Environment variable VECTOR_DB_ENABLE=true loaded correctly")
        else:
            print("❌ Environment variable not loaded correctly")
            return False
        
        # Test changing environment and reloading
        os.environ["VECTOR_DB_ENABLE"] = "false"
        
        from bu_processor.core.config import reload_config
        config2 = reload_config()
        
        if config2.vector_db.enable_vector_db == False:
            print("✅ Config reload with VECTOR_DB_ENABLE=false works")
        else:
            print("❌ Config reload failed")
            return False
        
        print("\n🎉 FINAL CONFIG TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if "VECTOR_DB_ENABLE" in os.environ:
            del os.environ["VECTOR_DB_ENABLE"]

if __name__ == "__main__":
    success = test_final_config()
    sys.exit(0 if success else 1)
