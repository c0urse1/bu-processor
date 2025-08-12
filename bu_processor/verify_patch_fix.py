#!/usr/bin/env python3
"""Verify the Pinecone/Pipeline patch fix by checking file content."""

import re
from pathlib import Path

def check_pipeline_imports():
    """Check if the necessary imports are present in enhanced_integrated_pipeline.py."""
    
    pipeline_file = Path("bu_processor/pipeline/enhanced_integrated_pipeline.py")
    
    if not pipeline_file.exists():
        print(f"❌ File not found: {pipeline_file}")
        return False
    
    content = pipeline_file.read_text(encoding='utf-8')
    
    checks = [
        ("PineconeManager import", r"from \.pinecone_integration import.*PineconeManager"),
        ("PineconeManager fallback class", r"class PineconeManager:"),
        ("BUProcessorChatbot import", r"from \.chatbot_integration import BUProcessorChatbot"),
        ("ChatbotIntegration alias", r"ChatbotIntegration = BUProcessorChatbot"),
        ("ChatbotIntegration fallback class", r"class ChatbotIntegration:"),
    ]
    
    all_passed = True
    
    print("🔍 Checking enhanced_integrated_pipeline.py for patch targets:")
    print("=" * 65)
    
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    return all_passed

def check_test_patches():
    """Check what classes the tests are trying to patch."""
    
    test_file = Path("tests/test_pipeline_components.py")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    content = test_file.read_text(encoding='utf-8')
    
    # Find all patch statements for enhanced_integrated_pipeline
    patch_patterns = re.findall(
        r'mocker\.patch\("bu_processor\.pipeline\.enhanced_integrated_pipeline\.(\w+)"',
        content
    )
    
    print(f"\n🧪 Classes that tests expect to patch:")
    print("=" * 65)
    
    expected_classes = {'PineconeManager', 'ChatbotIntegration'}
    found_classes = set(patch_patterns)
    
    for class_name in expected_classes:
        if class_name in found_classes:
            print(f"✅ Tests patch: bu_processor.pipeline.enhanced_integrated_pipeline.{class_name}")
        else:
            print(f"⚠️  Tests don't patch: bu_processor.pipeline.enhanced_integrated_pipeline.{class_name}")
    
    # Show any other classes being patched
    other_classes = found_classes - expected_classes
    if other_classes:
        print(f"\n📝 Other classes being patched:")
        for class_name in other_classes:
            print(f"   • {class_name}")
    
    return True

def main():
    """Main verification function."""
    print("🔧 VERIFYING PINECONE/PIPELINE PATCH FIX")
    print("=" * 70)
    
    check1 = check_pipeline_imports()
    check2 = check_test_patches()
    
    print(f"\n" + "=" * 70)
    print("📋 SUMMARY")  
    print("=" * 70)
    
    if check1 and check2:
        print("🎉 PATCH FIX VERIFICATION SUCCESSFUL!")
        print()
        print("✅ PineconeManager is now available for patching")
        print("✅ ChatbotIntegration is now available for patching")  
        print("✅ Fallback classes are defined for when imports fail")
        print()
        print("🔨 How the fix works:")
        print("   1. Imports PineconeManager from pinecone_integration.py")
        print("   2. Imports BUProcessorChatbot and aliases it as ChatbotIntegration")
        print("   3. Provides fallback dummy classes if imports fail")
        print("   4. Makes both classes available at module level for test patching")
        print()
        print("🎯 AttributeError should now be resolved!")
        print()
        print("Tests can now successfully patch:")
        print("  • bu_processor.pipeline.enhanced_integrated_pipeline.PineconeManager")
        print("  • bu_processor.pipeline.enhanced_integrated_pipeline.ChatbotIntegration")
        
        return True
    else:
        print("❌ PATCH FIX VERIFICATION FAILED")
        print("The fix needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
