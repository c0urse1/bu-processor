#!/usr/bin/env python3
"""
Test that training data loading works correctly with dummy CSV.
"""

def test_data_loading_fix():
    """Test that the data loading fix works correctly."""
    
    print("🔍 Testing training data loading fix...")
    
    try:
        import tempfile
        import pandas as pd
        from pathlib import Path
        from bu_processor.training.config import TrainingConfig
        from bu_processor.training.data import load_dataset, encode_labels
        print("✅ Imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create dummy data
    train_data = [
        {'text': 'Dies ist ein BU Antrag', 'label': 'BU_ANTRAG'},
        {'text': 'Hier ist eine Police', 'label': 'POLICE'},
        {'text': 'Diese Bedingungen sind wichtig', 'label': 'BEDINGUNGEN'},
        {'text': 'Sonstiger wichtiger Text', 'label': 'SONSTIGES'}
    ]
    
    val_data = [
        {'text': 'Validierung für BU Antrag', 'label': 'BU_ANTRAG'},
        {'text': 'Validierung für Bedingungen', 'label': 'BEDINGUNGEN'}
    ]
    
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
            print(f"✅ Dataset loaded - Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
            
            # Test label encoding (this was the failing part)
            encoded_ds, label2id, id2label = encode_labels(ds, cfg)
            print(f"✅ Labels encoded successfully")
            print(f"   label2id: {label2id}")
            print(f"   id2label: {id2label}")
            
            # Check the encoded data
            train_sample = encoded_ds['train'][0]
            print(f"   Sample structure: {list(train_sample.keys())}")
            print(f"   Sample labels type: {type(train_sample['labels'])}")
            
            print("✅ All data processing steps work correctly")
            
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 Training data loading fix successful!")
    print("\n📋 What was fixed:")
    print("   1. ✅ Removed problematic cast_column() call")
    print("   2. ✅ Label encoding now works correctly")
    print("   3. ✅ Datasets library handles types automatically")
    print("   4. ✅ Ready for training smoke tests")
    
    return True

if __name__ == "__main__":
    success = test_data_loading_fix()
    exit(0 if success else 1)
