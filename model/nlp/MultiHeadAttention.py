import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Self-attention layer compatible with PyTorch 1.0.0
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super(MultiHeadAttention, self).__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            attn_mask: Optional attention mask
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project and reshape to [batch_size, num_heads, seq_len, head_dim]
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights