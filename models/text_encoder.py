import torch
import torch.nn as nn
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Please install with: pip install git+https://github.com/openai/CLIP.git")


class TextEncoder(nn.Module):
    """
    CLIP-based text encoder for generating text embeddings.
    """
    def __init__(self, device='cuda', model_name='ViT-B/32'):
        super(TextEncoder, self).__init__()
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is not installed. Please install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = device
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        # Freeze CLIP model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        """
        Encode text into embeddings.
        
        Args:
            text: String or list of strings to encode
            
        Returns:
            text_features: Tensor of shape (batch_size, 512) for ViT-B/32
        """
        if isinstance(text, str):
            text = [text]
        
        tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features  # shape: (batch_size, 512)
    
    def encode_batch(self, text_list):
        """
        Encode a batch of text prompts.
        
        Args:
            text_list: List of text strings
            
        Returns:
            text_features: Tensor of shape (len(text_list), 512)
        """
        return self.forward(text_list)


