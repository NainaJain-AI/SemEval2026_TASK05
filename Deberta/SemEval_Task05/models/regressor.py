"""
Model architecture for T5-v1 AmbiStory word sense plausibility rating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, Tuple


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence."""
    
    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Hidden dimension of the encoder
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted pooling.
        
        Args:
            hidden_states: [batch, seq_len, hidden]
            attention_mask: [batch, seq_len]
            
        Returns:
            Tuple of (pooled output, attention weights)
        """
        attn_scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Mask padding tokens
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len]
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch, hidden]
        
        return pooled, attn_weights


class ImprovedSemEvalModel(nn.Module):
    """
    Enhanced model for SemEval 2026 Task 5.
    
    Features:
    - Attention pooling (not just CLS)
    - Multi-layer regression head with dropout and LayerNorm
    - GELU activation (better for transformers)
    - Combined CLS + attention pooling
    """
    
    def __init__(
        self,
        model_name: str,
        num_added_tokens: int = 0,
        dropout: float = 0.2
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., 'microsoft/deberta-v3-base')
            num_added_tokens: Number of special tokens added to tokenizer
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        # Resize embeddings if we added special tokens
        if num_added_tokens > 0:
            self.encoder.resize_token_embeddings(
                self.encoder.config.vocab_size + num_added_tokens
            )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden)
        
        # Combine CLS and attention pooling
        self.combine = nn.Linear(hidden * 2, hidden)
        
        # Deeper regression head
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize new layers with Xavier initialization."""
        for module in [self.attention_pool, self.combine, self.regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Predictions [batch] or (predictions, attention_weights)
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        # CLS token
        cls_output = hidden_states[:, 0]
        
        # Attention pooling
        attn_pooled, attn_weights = self.attention_pool(hidden_states, attention_mask)
        
        # Combine CLS and attention pooling
        combined = torch.cat([cls_output, attn_pooled], dim=-1)
        combined = self.combine(combined)
        combined = F.gelu(combined)
        
        # Regression
        logits = self.regressor(combined).squeeze(-1)
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def freeze_encoder(self, num_unfrozen_layers: int = 4):
        """
        Freeze encoder except last N layers.
        
        Args:
            num_unfrozen_layers: Number of top layers to keep unfrozen
        """
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        total_layers = self.encoder.config.num_hidden_layers
        unfrozen_layers = list(range(total_layers - num_unfrozen_layers, total_layers))
        
        for name, param in self.encoder.named_parameters():
            # Check if this parameter belongs to an unfrozen layer
            for layer_idx in unfrozen_layers:
                if f"layer.{layer_idx}" in name or f"layers.{layer_idx}" in name:
                    param.requires_grad = True
                    break
            
            # Always unfreeze embeddings resize if we added tokens
            if "word_embeddings" in name:
                param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
