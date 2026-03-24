import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLoRALinear(nn.Module):
    """Simplified LoRA linear layer implementation"""
    def __init__(
        self,
        in_features,
        out_features,
        r=8,
        lora_alpha=28,
        lora_dropout=0.3,
        bias=True,
        device=None,
        seed=None
    ):
        super().__init__()
        self.seed = seed  # Save seed attribute
        torch.manual_seed(seed)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Base linear layer parameters (not trained)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)
            
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros((r, in_features), device=device))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), device=device))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Set base weights to not require gradients
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
    def forward(self, x):
        # Base forward pass (frozen)
        base_output = F.linear(x, self.weight, self.bias)
        
        # Solve dropout randomness issue
        if self.training:
            # Use dropout during training
            torch.manual_seed(self.seed)
            lora_output = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        else:
            # Don't use dropout during testing, just scale
            lora_output = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1) * (1 - self.lora_dropout.p)

        # # LoRA path
        # lora_output = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        
        # Combine outputs
        return base_output + lora_output * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into base weights (for inference)"""
        if self.weight.requires_grad:
            return  # Already merged
        
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.weight.requires_grad = True
    
    def unmerge_weights(self):
        """Separate LoRA weights (restore training state)"""
        if not self.weight.requires_grad:
            return  # Already separated
            
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self.weight.requires_grad = False