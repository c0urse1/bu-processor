#!/usr/bin/env python3
"""
🧪 CONFIG SHIM TEST
================
Test that the config shim for enable_vector_db works properly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_shim():
    """Test that both vector_db_enable and enable_vector_db work."""
    print("🧪 CONFIG SHIM TEST")
    print("=" * 30)
    
    try:
        # Set environment variable for testing
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
        
        # Test both access methods
        try:
            field_value = vdb.vector_db_enable
            print(f"✅ vdb.vector_db_enable: {field_value}")
        except AttributeError as e:
            print(f"❌ vdb.vector_db_enable failed: {e}")
        
        try:
            property_value = vdb.enable_vector_db
            print(f"✅ vdb.enable_vector_db: {property_value}")
        except AttributeError as e:
            print(f"❌ vdb.enable_vector_db failed: {e}")
        
        # Test they're the same
        if hasattr(vdb, 'vector_db_enable') and hasattr(vdb, 'enable_vector_db'):
            if vdb.vector_db_enable == vdb.enable_vector_db:
                print("✅ Both access methods return same value")
            else:
                print("❌ Access methods return different values")
        
        print("\n🎉 CONFIG SHIM TEST COMPLETED!")
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
    success = test_config_shim()
    sys.exit(0 if success else 1)
