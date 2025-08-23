#!/usr/bin/env python3
"""
📋 MODEL INFORMATION SCRIPT
===========================

Queries the API /models/info endpoint to check available labels
and model configuration.

Uses BU_API_BASE and BU_API_TOKEN environment variables.
"""

import os
import requests
import json

# Configuration
API_BASE = os.getenv("BU_API_BASE", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("BU_API_TOKEN", "")

def main():
    """Query model information from API"""
    print(f"🔍 Checking model info at {API_BASE}")
    print(f"🔑 Using token: {'✅ Set' if API_TOKEN else '❌ Not set'}")
    print("=" * 50)
    
    # Prepare headers
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    
    try:
        # Query model info endpoint
        response = requests.get(
            f"{API_BASE}/models/info", 
            headers=headers, 
            timeout=10
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Pretty print the response
            print("✅ Model Information:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Extract key information
            if 'available_labels' in data:
                labels = data['available_labels']
                print(f"\n📋 Available Labels ({len(labels)}):")
                for i, label in enumerate(labels, 1):
                    print(f"   {i}. {label}")
            else:
                print("\n❌ No 'available_labels' found in response")
                
            if 'model_info' in data:
                model_info = data['model_info']
                print(f"\n🤖 Model Info:")
                print(f"   Model: {model_info.get('model', 'Unknown')}")
                print(f"   Labels available: {model_info.get('labels_available', 'Unknown')}")
            
        elif response.status_code == 401:
            print("❌ Authentication failed (401)")
            print("   Check BU_API_TOKEN environment variable")
        elif response.status_code == 403:
            print("❌ Access forbidden (403)")
            print("   Invalid API token")
        else:
            print(f"❌ Error {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Detail: {error_data}")
            except:
                print(f"   Raw response: {response.text}")
                
    except requests.ConnectionError:
        print("❌ Connection failed")
        print(f"   Is the API running at {API_BASE}?")
    except requests.Timeout:
        print("❌ Request timeout")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
