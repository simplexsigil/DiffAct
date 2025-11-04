#!/usr/bin/env python3
"""
Test script to verify omnifall data loading works correctly
"""

import os
import sys
import json
from dataset import get_data_dict, VideoFeatureDataset
from utils import load_config_file

def test_omnifall_loading():
    print("Testing OmniFall data loading...")
    
    # Load config
    config_path = "configs/OmniFall-Trained.json"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
        
    config = load_config_file(config_path)
    print(f"Loaded config: {config['dataset_name']}")
    
    # Set up paths like main.py does
    if config['dataset_name'] == 'omnifall':
        label2idx = {
            "walk": 0, "fall": 1, "fallen": 2, "sit_down": 3, "sitting": 4,
            "lie_down": 5, "lying": 6, "stand_up": 7, "standing": 8, "other": 9,
        }
        idx2label = {v: k for k, v in label2idx.items()}
        
        root_data_dir = config['root_data_dir']
        feature_dir = root_data_dir
        label_dir = os.path.join(root_data_dir, "segmentation_annotations", "labels", "full.csv")
        
        event_list = [idx2label[i] for i in range(len(label2idx))]
        num_classes = len(event_list)
        
        # Load CSV splits
        import pandas as pd
        train_df = pd.read_csv(os.path.join(root_data_dir, "segmentation_annotations", "splits", "train_cs.csv"))
        test_df = pd.read_csv(os.path.join(root_data_dir, "segmentation_annotations", "splits", "test_cs.csv"))
        
        train_video_list = train_df.iloc[:, 0].tolist()[:5]  # Test with first 5 videos
        test_video_list = test_df.iloc[:, 0].tolist()[:3]   # Test with first 3 videos
        
        print(f"Event list: {event_list}")
        print(f"Sample train videos: {train_video_list}")
        print(f"Sample test videos: {test_video_list}")
        
        # Test data loading
        try:
            print("\nTesting train data loading...")
            train_data_dict = get_data_dict(
                feature_dir=feature_dir,
                label_dir=label_dir,
                video_list=train_video_list,
                event_list=event_list,
                sample_rate=config['sample_rate'],
                temporal_aug=config['temporal_aug'],
                boundary_smooth=config['boundary_smooth']
            )
            
            print(f"Successfully loaded {len(train_data_dict)} train videos")
            
            # Test dataset creation
            train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
            print(f"Created train dataset with {len(train_dataset)} samples")
            
            # Test single sample
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                feature, label, boundary, video = sample
                print(f"Sample: video={video}, feature_shape={feature.shape}, label_shape={label.shape}")
                print(f"Feature dtype: {feature.dtype}, Label dtype: {label.dtype}")
                print(f"Label range: [{label.min():.2f}, {label.max():.2f}]")
                
            print("\nTesting test data loading...")
            test_data_dict = get_data_dict(
                feature_dir=feature_dir,
                label_dir=label_dir,
                video_list=test_video_list,
                event_list=event_list,
                sample_rate=config['sample_rate'],
                temporal_aug=config['temporal_aug'],
                boundary_smooth=config['boundary_smooth']
            )
            
            test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')
            print(f"Created test dataset with {len(test_dataset)} samples")
            
            if len(test_dataset) > 0:
                sample = test_dataset[0]
                features, label, boundaries, video = sample
                print(f"Test sample: video={video}, #feature_variants={len(features)}")
                if len(features) > 0:
                    print(f"Feature variant shape: {features[0].shape}")
                
            print("\n✅ All tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        print(f"This test is only for omnifall dataset, got: {config['dataset_name']}")
        return False

if __name__ == "__main__":
    success = test_omnifall_loading()
    sys.exit(0 if success else 1)