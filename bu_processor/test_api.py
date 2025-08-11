#!/usr/bin/env python3
"""
BU-Processor API Test Script
===========================

Einfaches Test-Script um die REST-API zu testen.
Führt verschiedene API-Aufrufe durch und zeigt die Ergebnisse an.
"""

import requests
import json
import time
import sys
from pathlib import Path

class APITester:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def test_health(self):
        """Test Health Check Endpoint"""
        print("🩺 Testing Health Check...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Status: {data['status']}")
                print(f"   📊 Version: {data['version']}")
                print(f"   🌍 Environment: {data['environment']}")
                print(f"   🤖 Classifier Loaded: {data['classifier_loaded']}")
                print(f"   ⏱️  Uptime: {data['uptime_seconds']:.1f}s")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_root(self):
        """Test Root Endpoint"""
        print("\n🏠 Testing Root Endpoint...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Message: {data['message']}")
                print(f"   📊 Version: {data['version']}")
                print(f"   📚 Docs: {data['docs']}")
                return True
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Root endpoint error: {e}")
            return False
    
    def test_text_classification(self):
        """Test Text Classification"""
        print("\n📝 Testing Text Classification...")
        test_text = "Ich arbeite als Softwareentwickler in einer großen IT-Firma und entwickle Web-Anwendungen."
        
        try:
            payload = {
                "text": test_text,
                "include_confidence": True,
                "include_processing_time": True
            }
            
            response = requests.post(
                f"{self.base_url}/classify/text",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Text: {test_text[:50]}...")
                print(f"   🎯 Category: {data['category']}")
                print(f"   🏷️  Label: {data.get('category_label', 'N/A')}")
                print(f"   📊 Confidence: {data['confidence']:.3f}")
                print(f"   ✅ Confident: {data['is_confident']}")
                if 'processing_time' in data:
                    print(f"   ⏱️  Processing Time: {data['processing_time']:.3f}s")
                return True
            else:
                print(f"   ❌ Text classification failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Text classification error: {e}")
            return False
    
    def test_batch_classification(self):
        """Test Batch Text Classification"""
        print("\n📦 Testing Batch Classification...")
        test_texts = [
            "Ich bin Arzt und arbeite im Krankenhaus.",
            "Als Lehrer unterrichte ich Mathematik an einer Schule.",
            "Ich arbeite als Marketing Manager in einem Startup.",
            "Meine Tätigkeit als Anwalt erfordert viel Recherche."
        ]
        
        try:
            payload = {
                "texts": test_texts,
                "batch_id": f"test_batch_{int(time.time())}"
            }
            
            response = requests.post(
                f"{self.base_url}/classify/batch",
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Batch ID: {data['batch_id']}")
                print(f"   📊 Total Processed: {data['total_processed']}")
                print(f"   ✅ Successful: {data['successful']}")
                print(f"   ❌ Failed: {data['failed']}")
                print(f"   ⏱️  Batch Time: {data['batch_time']:.3f}s")
                print(f"   ⚡ Avg Time per Text: {data['batch_time']/data['total_processed']:.3f}s")
                
                print(f"   📋 Results:")
                for i, result in enumerate(data['results'][:3]):  # Show first 3
                    print(f"      {i+1}. Category: {result['category']}, Confidence: {result['confidence']:.3f}")
                
                return True
            else:
                print(f"   ❌ Batch classification failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Batch classification error: {e}")
            return False
    
    def test_model_info(self):
        """Test Model Info Endpoint"""
        print("\n🤖 Testing Model Info...")
        try:
            response = requests.get(
                f"{self.base_url}/models/info",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                model_info = data.get('model_info', {})
                print(f"   ✅ Model Name: {model_info.get('model_name', 'N/A')}")
                print(f"   🏷️  Labels Available: {model_info.get('labels_available', False)}")
                print(f"   📊 Label Count: {model_info.get('label_count', 'N/A')}")
                print(f"   🖥️  Device: {model_info.get('device', 'N/A')}")
                print(f"   📦 Batch Size: {model_info.get('batch_size', 'N/A')}")
                
                strategies = data.get('supported_chunking_strategies', [])
                print(f"   🧩 Chunking Strategies: {', '.join(strategies)}")
                
                return True
            else:
                print(f"   ❌ Model info failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Model info error: {e}")
            return False
    
    def test_pdf_classification(self):
        """Test PDF Classification if test file exists"""
        print("\n📄 Testing PDF Classification...")
        
        # Look for test PDF files
        test_pdf_paths = [
            "tests/fixtures/sample.pdf",
            "../tests/fixtures/sample.pdf", 
            "sample.pdf"
        ]
        
        test_pdf = None
        for pdf_path in test_pdf_paths:
            if Path(pdf_path).exists():
                test_pdf = pdf_path
                break
        
        if not test_pdf:
            print("   ⚠️  No test PDF found, skipping PDF test")
            print(f"   💡 Looked for: {test_pdf_paths}")
            return None
        
        try:
            print(f"   📁 Using test PDF: {test_pdf}")
            
            with open(test_pdf, 'rb') as f:
                files = {'file': f}
                data = {
                    'chunking_strategy': 'simple',
                    'max_chunk_size': '1000',
                    'classify_chunks_individually': 'false'
                }
                
                # Don't include Content-Type for multipart
                headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}
                
                response = requests.post(
                    f"{self.base_url}/classify/pdf",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ File: {result['file_name']}")
                print(f"   🎯 Category: {result['category']}")
                print(f"   🏷️  Label: {result.get('category_label', 'N/A')}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
                print(f"   📄 Pages: {result['page_count']}")
                print(f"   📏 Text Length: {result['text_length']}")
                print(f"   🔧 Extraction Method: {result['extraction_method']}")
                return True
            else:
                print(f"   ❌ PDF classification failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ PDF classification error: {e}")
            return False

def main():
    """Main test function"""
    print("🚀 BU-Processor API Test Suite")
    print("=" * 50)
    
    # Parse command line arguments
    base_url = "http://localhost:8000"
    api_key = None
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    
    print(f"🎯 Testing API at: {base_url}")
    if api_key:
        print(f"🔑 Using API Key: {api_key[:10]}...")
    
    tester = APITester(base_url, api_key)
    
    # Run tests
    tests = [
        ("Health Check", tester.test_health),
        ("Root Endpoint", tester.test_root),
        ("Model Info", tester.test_model_info),
        ("Text Classification", tester.test_text_classification),
        ("Batch Classification", tester.test_batch_classification),
        ("PDF Classification", tester.test_pdf_classification)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"   💥 Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
            total += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            total += 1
        else:
            print(f"⚠️  {test_name}: SKIPPED")
    
    if total > 0:
        success_rate = (passed / total) * 100
        print(f"\n🎯 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 API is working well!")
        elif success_rate >= 50:
            print("⚠️  API has some issues")
        else:
            print("🚨 API has major problems")
    else:
        print("⚠️  No tests could be executed")
    
    print(f"\n💡 API Documentation: {base_url}/docs")
    print(f"🩺 Health Check: {base_url}/health")

if __name__ == "__main__":
    main()
