import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from .lora_layers import SimpleLoRALinear

class DeepfakeClassifier(nn.Module):
    def __init__(self, model, lora_rank=8, lora_alpha=28, lora_dropout=0.3, 
                 lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], seed=None):
        super().__init__()
        
        self.seed = seed  # Save seed for later use
        torch.manual_seed(seed)  # Set random seed

        self.model = model
        # Add ResNet for NPR feature processing
        from .restnet import ResNet, BasicBlock
        # Create custom ResNet for feature extraction - use only first two layers, simplified model
        self.npr_resnet = ResNet(BasicBlock, [2, 2], seed=seed)  # Use only first two layers, 512-dim output
        
        # More precisely detect feature dimensions
        with torch.no_grad():
            # Use small batch input to get actual feature dimensions
            device = next(model.parameters()).device
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            
            # Check NPR ResNet output dimension
            npr_dummy = torch.zeros(1, 3, 224, 224).to(device)
            npr_features = self.npr_resnet(npr_dummy)
            self.npr_feature_dim = npr_features.shape[-1]  # Should be 512
            print(f"NPR feature dimension: {self.npr_feature_dim}")
            
            # Extract vision features
            vision_outputs = model.vision_model(pixel_values=dummy_input)
            vision_features = vision_outputs.pooler_output
            self.vision_feature_dim = vision_features.shape[-1]
            print(f"Vision feature dimension: {self.vision_feature_dim}")
            
            # Get projected dimension
            projected_features = model.visual_projection(vision_features)
            self.projection_dim = projected_features.shape[-1]
            print(f"Projected feature dimension: {self.projection_dim}")
        
        # self.fc2 = nn.Linear(self.vision_feature_dim, self.npr_feature_dim)  # Correctly use detected dimensions
        self.fc2 =  nn.Sequential(
            nn.Linear(self.vision_feature_dim, self.npr_feature_dim),  # Map features to 128-dim
            nn.ReLU(),
            nn.Dropout(0.3),
            )
        self.fc1 = nn.Linear(self.npr_feature_dim, 2)

        # self.fusion_npr =  nn.Sequential(
        #     nn.Linear(self.npr_feature_dim * 2, self.npr_feature_dim),  # Map features to 128-dim
        #     nn.ReLU(),
        #     )

        # Add feature fusion layer - consider dimensions of both features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.npr_feature_dim + self.npr_feature_dim, self.npr_feature_dim),  # Map concatenated features back to 128-dim
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # LoRA configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        
        # Store references to replaced LoRA modules
        self.lora_layers = []
        
        # Freeze pretrained model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Apply LoRA
        self._apply_lora()
        
        # Output trainable parameter information
        self._print_trainable_parameters()
    
    def _apply_lora(self):
        """Apply LoRA to Vision Transformer attention layers"""
        # Clear previous LoRA layers
        self.lora_layers = []

        # Ensure random seed is set before creating LoRA layers
        torch.manual_seed(self.seed)
        
        # Iterate through and replace linear projection layers in attention modules
        for name, module in self.model.vision_model.named_modules():
            if "self_attn" in name:
                for target_name in self.lora_target_modules:
                    if hasattr(module, target_name):
                        # Get original linear layer
                        old_module = getattr(module, target_name)
                        if not isinstance(old_module, nn.Linear):
                            continue
                            
                        # Create LoRA layer replacement
                        lora_layer = SimpleLoRALinear(
                            seed=self.seed,  # Pass seed
                            in_features=old_module.in_features,
                            out_features=old_module.out_features,
                            r=self.lora_rank,
                            lora_alpha=self.lora_alpha,
                            lora_dropout=self.lora_dropout,
                            bias=old_module.bias is not None,
                            device=old_module.weight.device
                        )
                        
                        # Copy original weights
                        with torch.no_grad():
                            lora_layer.weight.copy_(old_module.weight)
                            if old_module.bias is not None and lora_layer.bias is not None:
                                lora_layer.bias.copy_(old_module.bias)
                        
                        # Replace module
                        setattr(module, target_name, lora_layer)
                        self.lora_layers.append(lora_layer)
    
    def _print_trainable_parameters(self):
        """Print trainable parameter statistics"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(
            f"Trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)"
        )

    def interpolate(self, img, factor):
        """Downsample then upsample to achieve image smoothing effect
        
        Args:
            img: Input image
            factor: Scaling factor
        """
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), 
            scale_factor=1/factor, mode='nearest', recompute_scale_factor=True
        )

    def forward(self, pixel_values, return_components=False):
        
        if not self.training and hasattr(self, 'seed') and self.seed is not None:
            torch.manual_seed(self.seed)

        """Model forward pass, enhanced with NPR features"""
        # 1. Calculate NPR difference features - pixel domain NPR calculation
        interpolated = self.interpolate(pixel_values, factor=0.5)
        npr_features = (pixel_values - interpolated)
        
        pixel_freq = fft.rfft2(pixel_values)                  
        interpolated_freq = fft.rfft2(interpolated)  
        freq_diff = (pixel_freq - interpolated_freq)
        combined_npr = torch.abs(npr_features + freq_diff ) 
        
        # Extract NPR features
        npr_features_resnet = self.npr_resnet(combined_npr) # [batch_size, 128]
        
        # 2. Original image path - use pretrained vision model to extract features
        orig_features = self.model.vision_model(pixel_values)
        orig_features = self.fc2(orig_features)

        fused_features = orig_features + npr_features_resnet
        
        # 3. Separate predictions - for monitoring individual contributions
        npr_logits = self.fc1(npr_features_resnet)
        lora_logits = self.fc1(orig_features)
        fusion_logits = self.fc1(fused_features)
        
        if return_components:
            return {
                'fusion_logits': fusion_logits,
                'lora_logits': lora_logits,
                'npr_logits': npr_logits,
                'lora_features': orig_features,
                'npr_features': npr_features_resnet
            }
        
        return fusion_logits
