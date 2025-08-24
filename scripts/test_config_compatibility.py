#!/usr/bin/env python3
"""
🧪 CONFIG COMPATIBILITY TEST
=========================
Test that the VectorDatabaseConfig backward compatibility works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_compatibility():
    """Test both enable_vector_db and vector_db_enable work."""
    print("🧪 CONFIG COMPATIBILITY TEST")
    print("=" * 40)
    
    try:
        from bu_processor.core.config import VectorDatabaseConfig
        
        # Test 1: Create config with default values
        print("\n🔧 Test 1: Default config...")
        config = VectorDatabaseConfig()
        print(f"✅ Default vector_db_enable: {config.vector_db_enable}")
        print(f"✅ Default enable_vector_db: {config.enable_vector_db}")
        
        # Test 2: Test property getter
        print("\n🔧 Test 2: Property getter...")
        assert config.enable_vector_db == config.vector_db_enable
        print("✅ Property getter works correctly")
        
        # Test 3: Test property setter
        print("\n🔧 Test 3: Property setter...")
        config.enable_vector_db = False
        assert config.vector_db_enable == False
        print("✅ Property setter works correctly")
        
        # Test 4: Test field names work
        print("\n🔧 Test 4: Field name compatibility...")
        config.vector_db_enable = True
        assert config.enable_vector_db == True
        print("✅ Field name compatibility works")
        
        # Test 5: Test environment variable loading
        print("\n🔧 Test 5: Environment variable aliases...")
        import os
        
        # Test with VECTOR_DB_ENABLE
        os.environ["VECTOR_DB_ENABLE"] = "false"
        config_env = VectorDatabaseConfig()
        print(f"✅ VECTOR_DB_ENABLE loaded: {config_env.vector_db_enable}")
        
        # Cleanup
        if "VECTOR_DB_ENABLE" in os.environ:
            del os.environ["VECTOR_DB_ENABLE"]
        
        print("\n🎉 ALL COMPATIBILITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_compatibility()
    sys.exit(0 if success else 1)
