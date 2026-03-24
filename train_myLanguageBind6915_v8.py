import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from transformers import CLIPModel, CLIPProcessor
from deepfake.utils import set_seed, worker_init_fn
from torch.utils.data import DataLoader, ConcatDataset 
from sklearn.metrics import accuracy_score, average_precision_score

from deepfake.deepfake_dataset import DeepfakeDetectionDataset
from deepfake.deepfake_classifier import DeepfakeClassifier

import logging
logging.basicConfig(level=logging.INFO)

# Configuration parameters
def parse_args():
    parser = argparse.ArgumentParser(description="LanguageBind Image LoRA Fine-tuning for Deepfake Detection")
    parser.add_argument("--dataset_dir", type=str, default="dataset_languagebind")
    parser.add_argument("--output_dir", type=str, default="myLanguageBind6915_v8", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)    
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initialization")
    parser.add_argument("--categories", type=str, default="car,cat,chair,horse", help="Dataset categories, comma separated")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--max_per_class", type=int, default=None, help="Limit maximum samples per class")
    
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lr_patience", type=int, default=3, help="Patience value for learning rate scheduler")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="Decay factor for learning rate scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    # Add data augmentation parameters
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    # parser.add_argument("--use_augmentation", type=bool, default=True, help="Use data augmentation")

    # Add print frequency parameter
    parser.add_argument("--print_freq", type=int, default=50, help="Print training info every N steps")

    # Add LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA adapters")
    parser.add_argument("--lora_alpha", type=float, default=28, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.3, help="Dropout rate for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,out_proj", help="Target modules for LoRA, comma separated")

    # Add differential training parameters
    parser.add_argument("--lora_lr", type=float, default=0.0002, 
                        help="Learning rate for LoRA parameters")
    parser.add_argument("--npr_lr", type=float, default=0.0002, 
                        help="Learning rate for NPR related parameters") 
    parser.add_argument("--fusion_lr", type=float, default=0.0002, 
                        help="Learning rate for Fusion layer parameters")
    
    return parser.parse_args()


# Phase 1 training function: joint training of LoRA and NPR
def train_joint(model, train_loader, criterion, optimizer, device, epoch, args):
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    
    for i, batch in enumerate(train_loader):        
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Get outputs from all components
        outputs = model(**inputs, return_components=True)
        fusion_logits = outputs['fusion_logits']
        lora_logits = outputs['lora_logits'] 
        npr_logits = outputs['npr_logits']

        # Calculate component losses
        fusion_loss = criterion(fusion_logits, labels)
        lora_loss = criterion(lora_logits, labels)
        npr_loss = criterion(npr_logits, labels)
        
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)

        # Use fusion loss for joint training phase
        # loss = fusion_loss
        loss = fusion_loss 
        loss.backward()
        optimizer.step()
        
        # Record training info
        current_loss = loss.item()
        train_loss += current_loss
        
        # Calculate predictions
        preds = torch.argmax(fusion_logits, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())
        
        # Display training info
        if i % args.print_freq == 0:
            fusion_lr = optimizer.param_groups[2]['lr']
    
    
    return train_loss, train_preds, train_labels


# Evaluation function
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    eval_preds, eval_labels = [], []
    eval_probs = []
    
    with torch.no_grad():
        for batch in eval_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            logits = model(**inputs)
            loss = criterion(logits, labels)
            
            eval_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get probability for fake class
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            eval_preds.extend(preds)
            eval_labels.extend(labels.cpu().numpy())
            eval_probs.extend(probs)
    
    # Calculate evaluation metrics
    eval_acc = accuracy_score(eval_labels, eval_preds)
    ap = average_precision_score(eval_labels, eval_probs)  # Calculate AP (Average Precision)
    
    return eval_loss, eval_acc, ap, eval_preds, eval_labels, eval_probs

test_seeds = [6915]

# Main function
def main():
    for seed in test_seeds:
        
        args = parse_args()
        args.seed = seed  # Set random seed
        set_seed(args.seed)  # Ensure seed is set as early as possible
        logging.info(f"Setting random seed to: {args.seed}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load LanguageBind image model and processor
        logging.info("Loading model...")
        # model = CLIPModel.from_pretrained("download/LanguageBind_Image")
        # Modified model loading
        model = CLIPModel.from_pretrained(
            "download/LanguageBind_Image",
            low_cpu_mem_usage=True,     # Save memory
            device_map=None,            # Don't use automatic device mapping
        )
        processor = CLIPProcessor.from_pretrained("download/LanguageBind_Image")
        
        # Print original model architecture info for debugging
        logging.info(f"CLIPModel vision architecture: {model.vision_model.__class__.__name__}")
        logging.info(f"CLIPModel vision projection: input dim={model.visual_projection.in_features}, output dim={model.visual_projection.out_features}")
        
        # Create classifier (add LoRA parameters)
        lora_target_modules = args.lora_target_modules.split(',')
        classifier = DeepfakeClassifier(
            model, 
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_target_modules,
            seed=args.seed
        )
        
        # Output trainable parameter information
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in classifier.parameters())
        logging.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
        
        # Parse category list
        categories = args.categories.split(',')
        logging.info(f"Training categories: {categories}")
        
        # Create datasets and data loaders
        train_datasets = []
        # eval_datasets = []
        
        for category in categories:
            logging.info(f"Loading {category} category data...")
            # Pass augmentation parameter when creating datasets
            train_dataset = DeepfakeDetectionDataset(
                args.dataset_dir, processor, split="train", category=category.strip(), 
                max_per_class=args.max_per_class, use_augmentation=args.use_augmentation,seed=args.seed
            )
            # eval_dataset = DeepfakeDetectionDataset(
            #     args.dataset_dir, processor, split="val", category=category.strip(), 
            #     max_per_class=args.max_per_class, use_augmentation=False, seed=args.seed
            # )
            
            train_datasets.append(train_dataset)
            # eval_datasets.append(eval_dataset)
        
        # Combine datasets from all categories
        combined_train_dataset = ConcatDataset(train_datasets)
        # combined_eval_dataset = ConcatDataset(eval_datasets)
        
        logging.info(f"Total training samples: {len(combined_train_dataset)}")
        # print(f"Total validation samples: {len(combined_eval_dataset)}")
        
        # train_loader = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # eval_loader = DataLoader(combined_eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        
        # Modified DataLoader creation
        train_loader = DataLoader(
            combined_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=args.seed),  # Correctly pass seed
            generator=torch.Generator().manual_seed(args.seed)  # Set generator seed
        )

        # eval_loader = DataLoader(
        #     combined_eval_dataset, 
        #     batch_size=args.batch_size, 
        #     shuffle=False,  # Don't shuffle validation set
        #     num_workers=args.num_workers,
        #     worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=args.seed),  # Correctly pass seed 
        #     generator=torch.Generator().manual_seed(args.seed)
        # )

        # Set device
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        classifier = classifier.to(device)
        
        # Simulate one forward pass to confirm dimension matching
        logging.info("Performing dimension check...")
        try:
            dummy_batch = next(iter(train_loader))
            dummy_inputs, _ = dummy_batch
            dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
            with torch.no_grad():
                _ = classifier(**dummy_inputs)
            logging.info("Dimension check passed ✓")
        except Exception as e:
            logging.info(f"Dimension check failed: {e}")
            import sys
            sys.exit(1)  # Terminate program early if dimension check fails
        
        # Separate parameter groups - LoRA parameters and NPR parameters
        lora_params = []
        npr_params = []
        fusion_params = []
        
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                if "lora" in name:
                    lora_params.append(param)
                elif "fusion" in name:  # Assume fusion-related layer params contain "fusion"
                    fusion_params.append(param)
                else:
                    npr_params.append(param)
        
        # Create optimizer for different parameter groups
        torch.manual_seed(args.seed)
        optimizer = optim.AdamW([
            {'params': lora_params, 'lr': args.lora_lr},
            {'params': npr_params, 'lr': args.npr_lr},
            {'params': fusion_params, 'lr': args.fusion_lr}
        ])
        
        criterion = nn.CrossEntropyLoss()
        
        # # Add learning rate scheduler
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='max',         # Use 'max' mode since we're monitoring accuracy
        #     factor=args.lr_factor, 
        #     patience=args.lr_patience, 
        #     verbose=True,
        #     min_lr=args.min_lr
        # )

        # Training loop
        
        for epoch in range(args.num_epochs):

            logging.info(f"Epoch {epoch+1}/{args.num_epochs} - Starting training")
            train_loss, train_preds, train_labels = train_joint(
                classifier, train_loader, criterion, optimizer, device, epoch, args
            )

            logging.info("Calculating training accuracy...")
            train_acc = accuracy_score(train_labels, train_preds)
            logging.info(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss/len(train_loader)}, Train Acc: {train_acc}")
            
            # # Evaluate model
            # print("Starting evaluation################################################################")
            # eval_loss, eval_acc, ap, _, _, _ = evaluate(classifier, eval_loader, criterion, device)
            
            # print(f"Epoch {epoch+1}/{args.num_epochs}, Eval Loss: {eval_loss/len(eval_loader)}, "
            #     f"Eval Acc: {eval_acc}, AP: {ap}")
            
            # # Adjust learning rate based on validation accuracy
            # scheduler.step(eval_acc)
            
            # Save current epoch model
            epoch_save_path = os.path.join(args.output_dir, f"model_seek_{args.seed}_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), epoch_save_path)
            logging.info(f"Saved epoch {epoch+1} model to {epoch_save_path}")

            # Test
            logging.info("Starting testing################################################################")
            test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "test_myLanguageBind6915_v8.py")
                
            test_command = f"CUDA_VISIBLE_DEVICES={3} python {test_script_path} --model_path {epoch_save_path} --seed {args.seed}"
            os.system(test_command)
            logging.info(f"Testing for seed {args.seed} completed")


if __name__ == "__main__":
    main()
