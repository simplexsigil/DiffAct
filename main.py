import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter


class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        epoch_pbar = tqdm(range(restore_epoch+1, num_epochs), desc="Training Progress")
        for epoch in epoch_pbar:

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(tqdm(train_train_loader, desc=f'Epoch {epoch}')):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            # Update epoch progress bar with basic info
            epoch_pbar.set_postfix({
                'Loss': f'{epoch_running_loss:.4f}',
                'Step': step
            })
            
            # Print epoch summary
            print(f'Epoch {epoch} - Training Loss: {epoch_running_loss:.4f}')
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    # Print test results per subset like MS-TCN2
                    print(f'\n=== Epoch {epoch} Test Results ===')
                    print(f"{'Dataset':<12} {'F1@10':<6} {'F1@25':<6} {'F1@50':<6} {'Edit':<6} {'Acc':<6} {'Videos':<6}")
                    print('-' * 60)
                    
                    # Access subset_results from the test method (need to modify return)
                    for subset_name, metrics in getattr(self, '_last_subset_results', {}).items():
                        if subset_name in ['all', 'OOPS']:
                            continue
                        print(f"{subset_name:<12} {metrics['F1@10']:<6.2f} {metrics['F1@25']:<6.2f} "
                              f"{metrics['F1@50']:<6.2f} {metrics['Edit']:<6.2f} {metrics['Acc']:<6.2f} {metrics['num_videos']:<6}")
                    
                    # Overall results (excluding OOPS)
                    if 'all' in getattr(self, '_last_subset_results', {}):
                        all_metrics = self._last_subset_results['all']
                        print('-' * 60)
                        print(f"{'Overall':<12} {all_metrics['F1@10']:<6.2f} {all_metrics['F1@25']:<6.2f} "
                              f"{all_metrics['F1@50']:<6.2f} {all_metrics['Edit']:<6.2f} {all_metrics['Acc']:<6.2f} {all_metrics['num_videos']:<6}")
                    
                    # OOPS-Fall results separately
                    if 'OOPS' in getattr(self, '_last_subset_results', {}):
                        oops_metrics = self._last_subset_results['OOPS']
                        print('-' * 60)
                        print(f"{'OOPS-Fall':<12} {oops_metrics['F1@10']:<6.2f} {oops_metrics['F1@25']:<6.2f} "
                              f"{oops_metrics['F1@50']:<6.2f} {oops_metrics['Edit']:<6.2f} {oops_metrics['Acc']:<6.2f} {oops_metrics['num_videos']:<6}")
                    
                    print('=' * 60)


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        # Print train results clearly
                        print(f'\n=== Epoch {epoch} Train Results ===')
                        print(f'Accuracy: {train_result_dict.get("Acc", 0):.2f}%')
                        print(f'Edit Score: {train_result_dict.get("Edit", 0):.2f}')
                        print(f'F1@10: {train_result_dict.get("F1@10", 0):.2f}%')
                        print(f'F1@25: {train_result_dict.get("F1@25", 0):.2f}%')
                        print(f'F1@50: {train_result_dict.get("F1@50", 0):.2f}%')
                        print('=' * 35)
                        
        if result_dir:
            logger.close()


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])


        self.model.eval()
        self.model.to(device)

        if model_path:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Loading from checkpoint file (latest.pt)
                self.model.load_state_dict(checkpoint['model'])
            else:
                # Loading from epoch model file (epoch-X.model)
                self.model.load_state_dict(checkpoint)

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed) 
                           for i in range(len(feature))] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()

            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Loading from checkpoint file (latest.pt)
                self.model.load_state_dict(checkpoint['model'])
            else:
                # Loading from epoch model file (epoch-X.model)
                self.model.load_state_dict(checkpoint)
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]
                
                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                
                # Create all necessary subdirectories
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        # Create ground truth directory for omnifall evaluation
        # Detect omnifall by checking if label_dir is a CSV file
        if label_dir.endswith('.csv'):
            gt_dir = os.path.join(result_dir, 'ground_truth')
            os.makedirs(gt_dir, exist_ok=True)
            
            # Create ground truth TXT files from the dataset
            for video_idx in range(len(test_dataset)):
                video = test_dataset.video_list[video_idx]
                # Get ground truth labels from dataset
                if test_dataset.mode == 'test':
                    gt_labels = test_dataset.data_dict[video]['event_seq_raw']
                    gt_labels = gt_labels.cpu().numpy().astype(int)
                    
                    # Convert to string labels
                    gt_strings = [self.event_list[int(label)] for label in gt_labels]
                    
                    # Save to TXT file
                    gt_file = os.path.join(gt_dir, f'{video}.txt')
                    os.makedirs(os.path.dirname(gt_file), exist_ok=True)
                    with open(gt_file, 'w') as f:
                        f.write('\n'.join(gt_strings) + '\n')
            
            eval_label_dir = gt_dir
        else:
            eval_label_dir = label_dir
            
        # Evaluate overall and per-subset like MS-TCN2
        test_subsets = [
            "all",
            "caucafall", 
            "cmdfall",
            "edf", 
            "gmdcsa24",
            "le2i",
            "mcfd",
            "occu", 
            "up_fall",
            "OOPS",
        ]
        
        subset_results = {}
        
        for subset in test_subsets:
            if subset == "all":
                # Exclude OOPS videos from overall statistics
                subset_videos = [v for v in test_dataset.video_list if 'OOPS' not in v]
            else:
                subset_videos = [v for v in test_dataset.video_list if subset in v]
            
            if len(subset_videos) == 0:
                continue
                
            acc, edit, f1s = func_eval(
                eval_label_dir, os.path.join(result_dir, 'prediction'), subset_videos)
            
            subset_results[subset] = {
                'Acc': acc,
                'Edit': edit,
                'F1@10': f1s[0],
                'F1@25': f1s[1],
                'F1@50': f1s[2],
                'num_videos': len(subset_videos)
            }

        # Store subset results for detailed printing
        self._last_subset_results = subset_results
        
        # Return overall results for logging
        result_dict = subset_results.get('all', {
            'Acc': 0, 'Edit': 0, 'F1@10': 0, 'F1@25': 0, 'F1@50': 0
        })
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--test-only', action='store_true', help='Skip training and only run testing')
    parser.add_argument('--model-path', type=str, help='Path to trained model file for testing')
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    # Omnifall-specific paths and labels
    if dataset_name == 'omnifall':
        label2idx = {
            "walk": 0,
            "fall": 1,
            "fallen": 2,
            "sit_down": 3,
            "sitting": 4,
            "lie_down": 5,
            "lying": 6,
            "stand_up": 7,
            "standing": 8,
            "other": 9,
        }
        idx2label = {v: k for k, v in label2idx.items()}
        
        # Override paths for omnifall
        root_data_dir = "/pfs/work8/workspace/ffhk/scratch/kf3609-ws/data/omnifall"
        feature_dir = root_data_dir  # Features are in subdirectories
        label_dir = os.path.join(root_data_dir, "segmentation_annotations", "labels", "full.csv")
        
        # Create event_list from label2idx (DiffAct expects this format)
        event_list = [idx2label[i] for i in range(len(label2idx))]
        num_classes = len(event_list)
    else:
        # Original code for other datasets
        feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
        label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
        mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')
        
        event_list = np.loadtxt(mapping_file, dtype=str)
        event_list = [i[1] for i in event_list]
        num_classes = len(event_list)

    if dataset_name == 'omnifall':
        # Load CSV splits for omnifall
        import pandas as pd
        train_df = pd.read_csv(os.path.join(root_data_dir, "segmentation_annotations", "splits", "train_cs.csv"))
        test_df = pd.read_csv(os.path.join(root_data_dir, "segmentation_annotations", "splits", "test_cs.csv"))
        
        train_video_list = train_df.iloc[:, 0].tolist()
        test_video_list = test_df.iloc[:, 0].tolist()
        
        # Check if OOPS videos are in training (they should only be in test)
        oops_in_train = [v for v in train_video_list if 'OOPS' in v]
        oops_in_test = [v for v in test_video_list if 'OOPS' in v]
        
        print(f"OOPS videos in training set: {len(oops_in_train)}")
        print(f"OOPS videos in test set: {len(oops_in_test)}")
        
        if oops_in_train:
            print("WARNING: OOPS videos found in training set:")
            for v in oops_in_train[:5]:  # Show first 5
                print(f"  - {v}")
            if len(oops_in_train) > 5:
                print(f"  ... and {len(oops_in_train) - 5} more")
        else:
            print("✓ OOPS videos are correctly excluded from training set")
    else:
        # Original bundle loading for other datasets
        train_video_list = np.loadtxt(os.path.join(
            root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
        test_video_list = np.loadtxt(os.path.join(
            root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

        train_video_list = [i.split('.')[0] for i in train_video_list]
        test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if args.test_only:
        # Test-only mode: load model and run testing
        if not args.model_path:
            raise ValueError("--model-path is required when using --test-only")
        
        print(f"Running test-only mode with model: {args.model_path}")
        test_result_dict = trainer.test(
            test_test_dataset, 'decoder-agg', 
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            label_dir, result_dir=os.path.join(result_dir, naming), 
            model_path=args.model_path
        )
        
        # Print results like during training
        print(f"\n=== Test Results ===")
        print(f"{'Dataset':<12} {'F1@10':<6} {'F1@25':<6} {'F1@50':<6} {'Edit':<6} {'Acc':<6} {'Videos':<6}")
        print('-' * 60)
        
        for subset_name, metrics in getattr(trainer, '_last_subset_results', {}).items():
            if subset_name in ['all', 'OOPS']:
                continue
            print(f"{subset_name:<12} {metrics['F1@10']:<6.2f} {metrics['F1@25']:<6.2f} "
                  f"{metrics['F1@50']:<6.2f} {metrics['Edit']:<6.2f} {metrics['Acc']:<6.2f} {metrics['num_videos']:<6}")
        
        # Overall results (excluding OOPS)
        if 'all' in getattr(trainer, '_last_subset_results', {}):
            all_metrics = trainer._last_subset_results['all']
            print('-' * 60)
            print(f"{'Overall':<12} {all_metrics['F1@10']:<6.2f} {all_metrics['F1@25']:<6.2f} "
                  f"{all_metrics['F1@50']:<6.2f} {all_metrics['Edit']:<6.2f} {all_metrics['Acc']:<6.2f} {all_metrics['num_videos']:<6}")
        
        # OOPS-Fall results separately
        if 'OOPS' in getattr(trainer, '_last_subset_results', {}):
            oops_metrics = trainer._last_subset_results['OOPS']
            print('-' * 60)
            print(f"{'OOPS-Fall':<12} {oops_metrics['F1@10']:<6.2f} {oops_metrics['F1@25']:<6.2f} "
                  f"{oops_metrics['F1@50']:<6.2f} {oops_metrics['Edit']:<6.2f} {oops_metrics['Acc']:<6.2f} {oops_metrics['num_videos']:<6}")
        
        print('=' * 60)
        
    else:
        # Normal training mode
        trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
            loss_weights, class_weighting, soft_label,
            num_epochs, batch_size, learning_rate, weight_decay,
            label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
            log_freq=log_freq, log_train_results=log_train_results
        )
