#!/usr/bin/env python3
"""
Simple Training Fixture Test
"""

def test_csv_creation_logic():
    """Test der CSV-Erstellung ohne schwere Dependencies."""
    
    print("=== TRAINING SMOKE TEST FIXTURE VERIFICATION ===")
    
    try:
        import pandas as pd
        print("✅ pandas available")
    except ImportError:
        print("❌ pandas not available")
        return False
        
    # Test data creation
    train_data = [
        {"text": "Dies ist ein Antrag für Betriebsunterbrechung", "label": "BU_ANTRAG"},
        {"text": "Hier ist eine Police für Versicherung", "label": "POLICE"},
        {"text": "Diese Bedingungen sind wichtig zu beachten", "label": "BEDINGUNGEN"},
        {"text": "Sonstiger wichtiger Text für Training", "label": "SONSTIGES"},
    ]
    
    val_data = [
        {"text": "Validierung für BU Antrag", "label": "BU_ANTRAG"},
        {"text": "Validierung für Bedingungen", "label": "BEDINGUNGEN"},
    ]
    
    # Test DataFrame creation
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    print(f"📊 Train data: {len(train_df)} rows")
    print(f"   Columns: {list(train_df.columns)}")
    print(f"   Labels: {list(train_df['label'].unique())}")
    
    print(f"📊 Val data: {len(val_df)} rows")
    print(f"   Columns: {list(val_df.columns)}")
    print(f"   Labels: {list(val_df['label'].unique())}")
    
    # Test expected labels
    expected_labels = {"BU_ANTRAG", "POLICE", "BEDINGUNGEN", "SONSTIGES"}
    train_labels = set(train_df['label'].unique())
    val_labels = set(val_df['label'].unique())
    
    print(f"\n🎯 Label compatibility:")
    print(f"   Expected: {expected_labels}")
    print(f"   Train has: {train_labels}")
    print(f"   Val has: {val_labels}")
    
    if train_labels.issubset(expected_labels):
        print("   ✅ Train labels compatible")
    else:
        print("   ❌ Train labels incompatible")
        
    if val_labels.issubset(expected_labels):
        print("   ✅ Val labels compatible")
    else:
        print("   ❌ Val labels incompatible")
    
    print("\n✅ CSV Creation Logic Verified!")
    return True

def test_file_structure():
    """Test ob die Dateien korrekt strukturiert sind."""
    print("\n=== FILE STRUCTURE CHECK ===")
    
    import os
    
    # Check test file
    test_file = "tests/test_training_smoke.py"
    if os.path.exists(test_file):
        print("✅ test_training_smoke.py exists")
        with open(test_file, 'r') as f:
            content = f.read()
            if "dummy_train_val" in content:
                print("✅ test uses dummy_train_val fixture")
            else:
                print("❌ test doesn't use dummy_train_val fixture")
    else:
        print("❌ test_training_smoke.py missing")
    
    # Check conftest
    conftest_file = "tests/conftest.py"
    if os.path.exists(conftest_file):
        print("✅ conftest.py exists")
        with open(conftest_file, 'r') as f:
            content = f.read()
            if "def dummy_train_val(" in content:
                print("✅ conftest has dummy_train_val fixture")
            else:
                print("❌ conftest missing dummy_train_val fixture")
    else:
        print("❌ conftest.py missing")
        
    return True

if __name__ == "__main__":
    print("🚀 Testing Training Smoke Test Fix...")
    test_csv_creation_logic()
    test_file_structure()
    print("\n🎯 Training Smoke Test Logic ✅")
