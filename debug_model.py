import torch
from transformers import CLIPModel, CLIPProcessor
from deepfake.deepfake_classifier import DeepfakeClassifier

def debug_model():
    print("Loading pretrained model...")
    # Modified model loading
    model = CLIPModel.from_pretrained(
        "download/LanguageBind_Image",
        low_cpu_mem_usage=True,     # Save memory
        device_map=None,            # Don't use automatic device mapping
    )
    processor = CLIPProcessor.from_pretrained("download/LanguageBind_Image")
    
    # Create classifier
    print("Creating classifier...")
    classifier = DeepfakeClassifier(
        model, 
        lora_rank=8,
        lora_alpha=28, 
        lora_dropout=0.3,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    
    # Create a random input
    print("Creating input...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        try:
            # Get NPR features
            interpolated = classifier.interpolate(dummy_input, scale_factor=0.5)
            npr_features = dummy_input - interpolated
            print(f"NPR feature shape: {npr_features.shape}")
            
            # Test ResNet
            from deepfake.restnet import ResNet, BasicBlock
            resnet = ResNet(BasicBlock, [2, 2], num_classes=512).to(device)
            npr_features_resnet = resnet(npr_features)
            print(f"ResNet output shape: {npr_features_resnet.shape}")
            
            # Test complete forward pass
            inputs = {"pixel_values": dummy_input}
            logits = classifier(**inputs)
            print(f"Model output shape: {logits.shape}")
            print("Test passed!")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_model()
