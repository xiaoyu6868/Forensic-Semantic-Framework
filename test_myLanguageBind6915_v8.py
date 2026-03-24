import os
import csv
import time
import torch
import random
import argparse
import numpy as np

from PIL import Image
from deepfake.utils import set_seed, worker_init_fn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from deepfake.deepfake_classifier import DeepfakeClassifier
from sklearn.metrics import accuracy_score, average_precision_score 

def parse_args():
    parser = argparse.ArgumentParser(description="LanguageBind Image Testing for Deepfake Detection")
    parser.add_argument("--model_path", type=str, default="myLanguageBind.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=28)
    parser.add_argument("--lora_dropout", type=float, default=0.3)
    return parser.parse_args()


# Function to recursively find image files
def find_images_recursively(root_dir, real_patterns, fake_patterns):
    real_images = []
    fake_images = []
    
    # Traverse all subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Check if current directory is a real image directory
        if any(pattern in root.split(os.path.sep) for pattern in real_patterns):
            for file in files:
                if file.lower().endswith((".jpg",".JPG",".jpeg",".JPEG",".png",".PNG",".bmp")):
                    real_images.append(os.path.join(root, file))
        
        # Check if current directory is a fake image directory
        if any(pattern in root.split(os.path.sep) for pattern in fake_patterns):
            for file in files:
                # if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if file.lower().endswith((".jpg",".JPG",".jpeg",".JPEG",".png",".PNG",".bmp")):
                    fake_images.append(os.path.join(root, file))
    
    return real_images, fake_images

# Dataset class
class DeepfakeTestDataset(Dataset):
    def __init__(self, dataroot, processor, no_resize=False, no_crop=True, seed=None):
        self.processor = processor
        
        self.seed = seed
        # Set internal randomness
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.dataroot = dataroot
        self.no_resize = no_resize
        self.no_crop = no_crop
        
        # Check if directory exists
        if not os.path.exists(dataroot):
            raise ValueError(f"Dataset path does not exist: {dataroot}")
        
        # Define patterns for real and fake images
        real_patterns = ['0_real', 'real', '0']
        fake_patterns = ['1_fake', 'fake', '1']
        
        # Recursively find all image files
        self.real_images, self.fake_images = find_images_recursively(dataroot, real_patterns, fake_patterns)
        
        # Ensure images were found
        if len(self.real_images) == 0 or len(self.fake_images) == 0:
            print(f"Warning: Not enough images found in path {dataroot}")
            print(f"Real images: {len(self.real_images)}, Fake images: {len(self.fake_images)}")
            # Continue execution, let errors occur in later processing
        
        # Combine real and fake image paths and labels
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0: real, 1: fake
        
        # Shuffle data
        combined = list(zip(self.images, self.labels))
        # random.shuffle(combined)
        random.Random(seed).shuffle(combined)  # Use fixed seed
        self.images, self.labels = zip(*combined)
        
        # print(f"Loaded {len(self.real_images)} real images and {len(self.fake_images)} fake images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process image using LanguageBind processor
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            
            return inputs, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a placeholder image and label
            return {"pixel_values": torch.zeros((3, 224, 224))}, torch.tensor(label, dtype=torch.long)

# Test function
def validate(model, processor, dataroot, no_resize=False, no_crop=True, batch_size=32, num_workers=4):
    try:
        # Create dataset and data loader
        args = parse_args()
        dataset = DeepfakeTestDataset(dataroot, processor, no_resize, no_crop, seed=args.seed)
        
        # Return early if dataset is empty
        if len(dataset) == 0:
            print(f"Warning: Dataset {dataroot} is empty")
            return 0.5, 0.5  # Return default values
            
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=args.seed),  # Fix this
        generator=torch.Generator().manual_seed(args.seed)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        # Make predictions - disable progress bar
        with torch.no_grad():
            # Remove tqdm progress bar
            for batch in dataloader:
                inputs, labels = batch
                # Ensure we only use pixel_values
                pixel_values = inputs["pixel_values"].to(device)
                
                # Use deepfake classifier directly
                logits = model(pixel_values)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Get fake class probability
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
        
        # Calculate evaluation metrics
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            acc = accuracy_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_probs)
        else:
            print("Warning: Insufficient or single-class labels, can't calculate accurate metrics")
            acc = 0.5
            ap = 0.5
        
        return acc, ap
    except Exception as e:
        print(f"Validation process error: {str(e)}")
        return 0.5, 0.5  # Return default values


def main():
    args = parse_args()
    
    # Set random seed
    print(f"Setting random seed to: {args.seed}")
    set_seed(args.seed)
    
    # Define datasets to test
    DetectionTests = {
        'DiffusionForensics': { 
            'dataroot': 'dataset_languagebind/test/DiffusionForensics/',
            'no_resize': False,
            'no_crop': True,
        },
        'Diffusion1kStep': { 
                    'dataroot': 'dataset_languagebind/test/Diffusion1kStep/',
                    'no_resize': False,
                    'no_crop': True,
                },
        'ForenSynths': { 
            'dataroot': 'dataset_languagebind/test/ForenSynths/',
            'no_resize': False,
            'no_crop': True,
        },
        'GANGen-Detection': { 
            'dataroot': 'dataset_languagebind/test/GANGen-Detection/',
            'no_resize': True,
            'no_crop': True,
        },
        
        'UniversalFakeDetect': { 
            'dataroot': 'dataset_languagebind/test/UniversalFakeDetect/',
            'no_resize': False,
            'no_crop': True,
        },
        
    }
    
    # Load LanguageBind model and processor
    print("Loading LanguageBind model...")
    # base_model = CLIPModel.from_pretrained("download/LanguageBind_Image")
    # Modified model loading
    base_model = CLIPModel.from_pretrained(
        "download/LanguageBind_Image",
        low_cpu_mem_usage=True,     # Save memory
        device_map=None,            # Don't use automatic device mapping
    )
    processor = CLIPProcessor.from_pretrained("download/LanguageBind_Image")

    # Create classifier
    model = DeepfakeClassifier(
        base_model,  # Use previously defined base_model
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        lora_target_modules=["q_proj","k_proj", "v_proj", "out_proj"],
        seed=args.seed
    )
    
    # Load trained model
    print(f"Model_path {args.model_path}")
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # Process state_dict if needed
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Remove possible "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    # Only load classifier part weights
    classifier_state_dict = {}
    for k, v in new_state_dict.items():
        # Look for fc1, fc2 and LoRA related layer weights
        if (k.startswith('fc1.') or k.startswith('fc2.') or 
            k.startswith('lora_') or k.startswith('npr_') or
            k.startswith('npr_resnet.') or k.startswith('npr_feature_adapter.') or
            k.startswith('feature_fusion.')):
            classifier_state_dict[k] = v
        # if k.startswith('classifier.'):
        #     classifier_state_dict[k] = v
    
    if len(classifier_state_dict) > 0:
        # Load model
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded complete model weights")
        except Exception as e:
            print(f"Failed to load complete model: {str(e)}")
            try:
                # Try to load just the classifier part
                missing_keys, unexpected_keys = model.load_state_dict(classifier_state_dict, strict=False)
                print(f"Loaded classifier weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            except Exception as e2:
                print(f"Failed to load classifier weights too: {str(e2)}")
    else:
        print("Warning: No classifier weights found, using uninitialized model")
    
    # Record all test results
    results = []
    
    # Test each dataset
    for testSet in DetectionTests.keys():
        print(f"\n{'='*33}\n{testSet:^33}\n{'='*33}")
        dataroot = DetectionTests[testSet]['dataroot']
        no_resize = DetectionTests[testSet]['no_resize']
        no_crop = DetectionTests[testSet]['no_crop']
        
        accs = []
        aps = []
        
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        
        # Test each subdataset
        for v_id, val in enumerate(sorted(os.listdir(dataroot))):
            sub_dataroot = os.path.join(dataroot, val)
            
            # Check if it's a valid directory
            if not os.path.isdir(sub_dataroot):
                continue
            
            try:
                acc, ap = validate(
                    model, 
                    processor, 
                    sub_dataroot, 
                    no_resize=no_resize, 
                    no_crop=no_crop,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
                
                accs.append(acc)
                aps.append(ap)
                
                print(f"({v_id} {val:<12}) acc: {acc*100:.1f}; ap: {ap*100:.1f}")
                
                # Save results
                results.append({
                    'dataset': testSet,
                    'subset': val,
                    'acc': acc,
                    'ap': ap
                })
                
            except Exception as e:
                print(f"Error testing subdataset {val}: {str(e)}")
        
        # Calculate mean if there are valid results
        if accs:
            mean_acc = np.array(accs).mean()
            mean_ap = np.array(aps).mean()
            print(f"({len(accs)} Mean      ) acc: {mean_acc*100:.1f}; ap: {mean_ap*100:.1f}")
            
            # Save mean results
            results.append({
                'dataset': testSet,
                'subset': 'Mean',
                'acc': mean_acc,
                'ap': mean_ap
            })
        
        print('*' * 25)

if __name__ == "__main__":
    main() 