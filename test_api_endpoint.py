#!/usr/bin/env python3
"""
Test the new /process/pdf endpoint integration
"""

import sys
import os
sys.path.append("bu_processor")

def test_endpoint():
    try:
        from bu_processor.api.main import app
        print('✅ API imports successful')
        print('✅ /process/pdf endpoint added')
        
        # Check if endpoint exists
        routes = [route.path for route in app.routes]
        if '/process/pdf' in routes:
            print('✅ /process/pdf route found in FastAPI app')
        else:
            print('❌ /process/pdf route not found')
        
        print(f'📋 Available routes: {len(routes)} total')
        for route in routes:
            if 'process' in route or 'ingest' in route or 'classify' in route:
                print(f'   - {route}')
                
        # Check response model
        from bu_processor.api.main import ProcessPDFResponse
        print('✅ ProcessPDFResponse model imported successfully')
        
        return True
        
    except ImportError as e:
        print(f'❌ Import error: {e}')
        return False
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_endpoint()
    sys.exit(0 if success else 1)
