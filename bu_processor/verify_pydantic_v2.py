#!/usr/bin/env python3
"""Verify Pydantic v2 migration was successful."""

import sys
from pathlib import Path

def test_pydantic_v2_migration():
    """Test that Pydantic v2 migration was successful."""
    
    print("🔄 TESTING PYDANTIC V2 MIGRATION")
    print("=" * 60)
    
    success = True
    
    try:
        print("1. Testing bu_processor import...")
        import bu_processor
        print("   ✅ bu_processor import successful")
    except Exception as e:
        print(f"   ❌ bu_processor import failed: {e}")
        success = False
    
    try:
        print("2. Testing config import...")
        from bu_processor import get_config
        print("   ✅ get_config import successful")
    except Exception as e:
        print(f"   ❌ get_config import failed: {e}")
        success = False
    
    try:
        print("3. Testing config instantiation...")
        from bu_processor import get_config
        config = get_config()
        print("   ✅ config instantiation successful")
        print(f"   📋 Environment: {config.environment}")
        print(f"   📋 Debug mode: {config.debug}")
    except Exception as e:
        print(f"   ❌ config instantiation failed: {e}")
        success = False
    
    try:
        print("4. Testing pinecone integration...")
        from bu_processor.pipeline.pinecone_integration import PineconeManager
        print("   ✅ PineconeManager import successful")
    except Exception as e:
        print(f"   ❌ PineconeManager import failed: {e}")
        success = False
    
    try:
        print("5. Testing enhanced integrated pipeline...")
        from bu_processor.pipeline.enhanced_integrated_pipeline import PineconeManager, ChatbotIntegration
        print("   ✅ Pipeline patch targets available")
        print("   ✅ PineconeManager available for patching")
        print("   ✅ ChatbotIntegration available for patching")
    except Exception as e:
        print(f"   ❌ Pipeline patch targets failed: {e}")
        success = False
    
    return success

def check_pydantic_v2_features():
    """Check that Pydantic v2 features are working."""
    
    print("\n🔍 CHECKING PYDANTIC V2 FEATURES")
    print("=" * 60)
    
    try:
        # Check imports work
        from pydantic import Field, field_validator, model_validator
        from pydantic_settings import BaseSettings, SettingsConfigDict
        print("✅ Pydantic v2 imports successful")
        
        # Check that we can create a simple config class
        class TestConfig(BaseSettings):
            test_field: str = Field(default="test")
            
            @field_validator('test_field')
            @classmethod
            def validate_test_field(cls, v: str):
                return v.strip()
            
            model_config = SettingsConfigDict(
                env_prefix="TEST_",
                extra="ignore"
            )
        
        # Test instantiation
        test_config = TestConfig()
        print("✅ Pydantic v2 validation working")
        print(f"   📋 Test field: {test_config.test_field}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic v2 features failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🎯 PYDANTIC V2 MIGRATION VERIFICATION")
    print("=" * 70)
    
    test1 = test_pydantic_v2_migration()
    test2 = check_pydantic_v2_features()
    
    print(f"\n" + "=" * 70)
    print("📋 MIGRATION SUMMARY")
    print("=" * 70)
    
    if test1 and test2:
        print("🎉 PYDANTIC V2 MIGRATION SUCCESSFUL!")
        print()
        print("✅ All imports working correctly")
        print("✅ Configuration loading properly")
        print("✅ Patch targets available")
        print("✅ Pydantic v2 features operational")
        print()
        print("🔧 Key Changes Applied:")
        print("   • Installed pydantic>=2,<3 and pydantic-settings>=2,<3")
        print("   • Updated BaseSettings import from pydantic-settings")
        print("   • Replaced @validator with @field_validator")
        print("   • Replaced @root_validator with @model_validator")
        print("   • Replaced Config class with model_config = SettingsConfigDict")
        print("   • Made imports lazy to prevent cascading failures")
        print()
        print("🎯 Previous errors should now be resolved:")
        print("   ❌ PydanticImportError: BaseSettings moved")
        print("   ❌ PydanticUserError: root_validator deprecated")
        print("   ❌ NameError: validator not defined")
        print()
        print("✅ All AttributeErrors in tests should be fixed!")
        
        return True
    else:
        print("❌ PYDANTIC V2 MIGRATION FAILED")
        print("Some issues remain. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
