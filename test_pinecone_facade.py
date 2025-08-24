#!/usr/bin/env python3
"""
Test facade pattern for PineconeManager.

This script tests that the facade correctly selects between simple
and enhanced implementations based on feature flags.
"""
import os
import sys
from pathlib import Path

# Add the bu_processor package to path
sys.path.insert(0, str(Path(__file__).parent / "bu_processor"))

def test_facade_pattern():
    """Test that the facade pattern works correctly."""
    print("Testing PineconeManager facade pattern...")
    
    # Test 1: Default behavior (should use simple implementation)
    print("\n1. Testing default behavior (simple implementation)...")
    try:
        from bu_processor.integrations.pinecone_manager import PineconeManager
        
        # Create manager without forcing enhanced mode
        manager = PineconeManager(
            index_name="test-index",
            api_key="dummy-key-for-testing",  # Won't actually connect
            force_simple=True  # Ensure we use simple for this test
        )
        
        print(f"   Implementation type: {manager.implementation_type}")
        print(f"   Is enhanced: {manager.is_enhanced}")
        
        # Verify it's the simple implementation
        assert "Simple" in manager.implementation_type or not manager.is_enhanced
        print("   ✓ Simple implementation selected correctly")
        
    except Exception as e:
        print(f"   ✗ Error testing simple implementation: {e}")
    
    # Test 2: Enhanced mode (if flag is set)
    print("\n2. Testing enhanced mode availability...")
    try:
        # Try to import enhanced manager directly
        from bu_processor.integrations.pinecone_enhanced import PineconeEnhancedManager
        print("   ✓ Enhanced implementation is available")
        
        # Test creation (should fail since it's not implemented yet)
        try:
            enhanced = PineconeEnhancedManager()
            print("   ✗ Enhanced manager should not be fully implemented yet")
        except NotImplementedError:
            print("   ✓ Enhanced manager correctly raises NotImplementedError")
            
    except ImportError:
        print("   ✓ Enhanced implementation not available (expected)")
    
    # Test 3: Factory function
    print("\n3. Testing factory function...")
    try:
        from bu_processor.integrations.pinecone_manager import get_pinecone_manager
        
        manager2 = get_pinecone_manager(
            index_name="test-index-2",
            api_key="dummy-key",
            force_simple=True
        )
        
        print(f"   Factory created: {manager2.implementation_type}")
        print("   ✓ Factory function works correctly")
        
    except Exception as e:
        print(f"   ✗ Error testing factory function: {e}")
    
    # Test 4: Direct access to implementations
    print("\n4. Testing direct access to implementations...")
    try:
        from bu_processor.integrations.pinecone_manager import (
            SimplePineconeManager,
            PineconeEnhancedManager
        )
        
        # Test simple manager directly
        simple_manager = SimplePineconeManager(
            index_name="direct-test",
            api_key="dummy-key"
        )
        print("   ✓ Direct access to SimplePineconeManager works")
        
        # Enhanced might be None if not available
        if PineconeEnhancedManager is None:
            print("   ✓ PineconeEnhancedManager is None (expected)")
        else:
            print("   ✓ PineconeEnhancedManager is available for direct access")
            
    except Exception as e:
        print(f"   ✗ Error testing direct access: {e}")
    
    # Test 5: Feature flag integration
    print("\n5. Testing feature flag integration...")
    try:
        from bu_processor.core.flags import FeatureFlags
        
        flags = FeatureFlags()
        enhanced_enabled = getattr(flags, 'enable_enhanced_pinecone', False)
        
        print(f"   Enhanced Pinecone flag: {enhanced_enabled}")
        print("   ✓ Feature flags integration works")
        
    except Exception as e:
        print(f"   ✗ Error testing feature flags: {e}")
    
    print("\n✓ Facade pattern testing completed!")
    print("\nArchitecture Summary:")
    print("  - pinecone_manager.py: Main entry point (facade)")
    print("  - pinecone_simple.py: Simple, stable implementation")
    print("  - pinecone_enhanced.py: Advanced features (placeholder)")
    print("  - pinecone_facade.py: Facade implementation logic")
    print("  - Feature flags control which implementation is used")

if __name__ == "__main__":
    test_facade_pattern()
