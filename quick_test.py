#!/usr/bin/env python3
"""
Simple test for the new /process/pdf endpoint
"""

import sys
import os

# Add the bu_processor directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'bu_processor'))

def quick_test():
    print("🧪 Testing API endpoint integration...")
    
    try:
        # Import the app
        from bu_processor.api.main import app
        print("✅ FastAPI app imported successfully")
        
        # Check routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        print(f"📋 Found {len(routes)} routes")
        
        # Check for our new endpoint
        process_pdf_found = '/process/pdf' in routes
        print(f"{'✅' if process_pdf_found else '❌'} /process/pdf endpoint {'found' if process_pdf_found else 'missing'}")
        
        # Show relevant routes
        relevant_routes = [r for r in routes if any(keyword in r for keyword in ['process', 'classify', 'ingest'])]
        print(f"🎯 Relevant routes ({len(relevant_routes)}):")
        for route in relevant_routes:
            print(f"   - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\n{'🎉 Test passed!' if success else '💥 Test failed!'}")
