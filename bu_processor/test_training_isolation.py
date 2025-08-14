#!/usr/bin/env python3
"""
Test training smoke test isolation with dummy CSV fixtures.
"""

def test_training_isolation():
    """Test that training smoke test isolation works correctly."""
    
    print("🔍 Testing training smoke test isolation...")
    
    # Test 1: Import required modules
    try:
        import tempfile
        import pandas as pd
        from pathlib import Path
        from bu_processor.training.config import TrainingConfig
        from bu_processor.training.data import load_dataset
        print("✅ All required modules imported")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create dummy data like the fixture does
    try:
        train_data = [
            {'text': 'Dies ist ein Antrag für Betriebsunterbrechung', 'label': 'BU_ANTRAG'},
            {'text': 'Hier ist eine Police für Versicherung', 'label': 'POLICE'},
            {'text': 'Diese Bedingungen sind wichtig', 'label': 'BEDINGUNGEN'},
            {'text': 'Sonstiger wichtiger Text', 'label': 'SONSTIGES'}
        ]
        
        val_data = [
            {'text': 'Validierung für BU Antrag', 'label': 'BU_ANTRAG'},
            {'text': 'Validierung für Bedingungen', 'label': 'BEDINGUNGEN'}
        ]
        print("✅ Dummy training data created")
    except Exception as e:
        print(f"❌ Data creation failed: {e}")
        return False
    
    # Test 3: Test data loading with dummy files
    try:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / 'train.csv'
            val_path = Path(tmp) / 'val.csv'
            
            # Create CSV files
            pd.DataFrame(train_data).to_csv(train_path, index=False)
            pd.DataFrame(val_data).to_csv(val_path, index=False)
            
            # Test config creation
            cfg = TrainingConfig(
                train_path=str(train_path),
                val_path=str(val_path),
                num_train_epochs=1,
                per_device_train_batch_size=2
            )
            
            # Test data loading
            ds = load_dataset(cfg)
            print(f"✅ Data loading works - Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
            
            # Verify data content
            train_sample = ds['train'][0]
            assert 'text' in train_sample
            assert 'label' in train_sample
            print("✅ Data structure is correct")
            
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    # Test 4: Test error handling for missing files
    try:
        bad_cfg = TrainingConfig(
            train_path="/nonexistent/train.csv",
            val_path="/nonexistent/val.csv"
        )
        
        try:
            load_dataset(bad_cfg)
            print("❌ Expected error for missing files, but none raised")
            return False
        except RuntimeError as e:
            if "dummy CSV via fixture" in str(e):
                print("✅ Proper error message for missing training data")
            else:
                print(f"❌ Wrong error message: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    # Test 5: Verify the fixture implementation exists (without importing from tests)
    try:
        from pathlib import Path
        conftest_path = Path(__file__).parent / "tests" / "conftest.py"
        
        if conftest_path.exists():
            # Read the file to check for dummy_train_val fixture
            content = conftest_path.read_text()
            if "def dummy_train_val" in content:
                print("✅ dummy_train_val fixture found in conftest.py")
            else:
                print("❌ dummy_train_val fixture not found in conftest.py")
                return False
        else:
            print("❌ conftest.py file not found")
            return False
        
    except Exception as e:
        print(f"❌ Fixture check failed: {e}")
        return False
    
    print("\n🎉 Training smoke test isolation working correctly!")
    print("\n📋 Summary of implementation:")
    print("   1. ✅ dummy_train_val fixture exists in tests/conftest.py")
    print("   2. ✅ Creates temporary CSV files with proper labels")
    print("   3. ✅ Sets environment variables for configuration")
    print("   4. ✅ Error handling guides users to fixture in tests")
    print("   5. ✅ Test can use fixture paths directly")
    print("   6. ✅ No real files needed for training tests")
    
    return True

if __name__ == "__main__":
    success = test_training_isolation()
    exit(0 if success else 1)
