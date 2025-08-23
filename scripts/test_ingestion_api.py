#!/usr/bin/env python3
"""
🧪 API TESTING SCRIPT FOR BACKGROUND INGESTION
==============================================

Tests the background PDF ingestion API endpoints with comprehensive scenarios.
"""

import requests
import time
import json
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key-123"  # Should match your .env configuration

def test_api_health():
    """Test API health endpoint"""
    print("🩺 Testing API Health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Environment: {data.get('environment')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_background_ingestion():
    """Test background PDF ingestion"""
    print("\n📄 Testing Background PDF Ingestion...")
    
    # Create a test PDF file
    test_pdf_path = Path("test_document.pdf")
    
    if not test_pdf_path.exists():
        print(f"❌ Test PDF file not found: {test_pdf_path}")
        print("Please create a test PDF file or use an existing one")
        return False
    
    try:
        # Upload PDF for background processing
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        with open(test_pdf_path, "rb") as f:
            files = {"file": (test_pdf_path.name, f, "application/pdf")}
            
            print(f"📤 Uploading {test_pdf_path.name}...")
            response = requests.post(
                f"{API_BASE_URL}/ingest/pdf",
                files=files,
                headers=headers
            )
        
        if response.status_code == 200:
            data = response.json()
            job_id = data.get("job_id")
            
            print(f"✅ Upload successful!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {data.get('status')}")
            print(f"   Tracking URL: {data.get('tracking_url')}")
            
            # Monitor job progress
            return monitor_job_progress(job_id, headers)
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Background ingestion test failed: {e}")
        return False

def monitor_job_progress(job_id: str, headers: dict, timeout: int = 300):
    """Monitor job progress until completion"""
    print(f"\n⏳ Monitoring job progress: {job_id}")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{API_BASE_URL}/ingest/status/{job_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                print(f"📊 Job Status: {status}")
                
                if status == "completed":
                    print("✅ Job completed successfully!")
                    
                    # Show results
                    result = data.get("result", {})
                    classification = result.get("classification", {})
                    
                    print(f"🏷️  Classification Results:")
                    print(f"   Predicted Label: {classification.get('predicted_label')}")
                    print(f"   Confidence: {classification.get('confidence'):.3f}")
                    print(f"   Text Length: {result.get('text_length')}")
                    
                    if data.get("processing_duration"):
                        print(f"⏱️  Processing Duration: {data.get('processing_duration'):.2f}s")
                    
                    return True
                
                elif status == "failed":
                    print(f"❌ Job failed!")
                    if data.get("error_message"):
                        print(f"   Error: {data.get('error_message')}")
                    return False
                
                elif status in ["pending", "running", "retrying"]:
                    if status == "retrying":
                        print(f"🔄 Job is retrying (attempt {data.get('retry_count', 0)})")
                    time.sleep(5)  # Wait 5 seconds before checking again
                    continue
                
            else:
                print(f"❌ Failed to get job status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            time.sleep(5)
    
    print(f"⏰ Timeout waiting for job completion")
    return False

def test_job_listing():
    """Test job listing endpoint"""
    print("\n📋 Testing Job Listing...")
    
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"{API_BASE_URL}/ingest/jobs", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            total_jobs = data.get("total_jobs", 0)
            
            print(f"✅ Job listing successful!")
            print(f"   Total Jobs: {total_jobs}")
            
            if total_jobs > 0:
                print("📄 Recent Jobs:")
                for job in data.get("jobs", [])[:3]:  # Show first 3 jobs
                    print(f"   - {job.get('job_id')[:8]}... | {job.get('filename')} | {job.get('status')}")
            
            return True
        else:
            print(f"❌ Job listing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Job listing test failed: {e}")
        return False

def test_text_classification():
    """Test direct text classification endpoint"""
    print("\n📝 Testing Direct Text Classification...")
    
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_text = """
        Dies ist ein Testdokument für die Klassifikation.
        Es enthält wichtige Geschäftsinformationen und sollte
        entsprechend kategorisiert werden.
        """
        
        payload = {
            "text": test_text.strip(),
            "include_confidence": True,
            "include_processing_time": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/classify/text",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Text classification successful!")
            print(f"   Category: {data.get('category')}")
            print(f"   Label: {data.get('category_label')}")
            print(f"   Confidence: {data.get('confidence'):.3f}")
            print(f"   Is Confident: {data.get('is_confident')}")
            
            if data.get("processing_time"):
                print(f"   Processing Time: {data.get('processing_time'):.3f}s")
            
            return True
        else:
            print(f"❌ Text classification failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text classification test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 BU-Processor API Testing Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test API health
    test_results.append(("API Health", test_api_health()))
    
    # Test text classification
    test_results.append(("Text Classification", test_text_classification()))
    
    # Test background ingestion (if test file exists)
    if Path("test_document.pdf").exists() or Path("test_bu_document.pdf").exists():
        test_results.append(("Background Ingestion", test_background_ingestion()))
    else:
        print("\n⚠️  No test PDF found, skipping background ingestion test")
        print("   Create 'test_document.pdf' or 'test_bu_document.pdf' to test ingestion")
    
    # Test job listing
    test_results.append(("Job Listing", test_job_listing()))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the API configuration and logs.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
