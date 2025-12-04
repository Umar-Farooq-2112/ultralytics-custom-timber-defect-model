# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom MobileNetV3-based blocks for lightweight YOLO models."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

__all__ = (
    "DWConvCustom",
    "ConvBNAct",
    "CBAM_ChannelOnly",
    "SimSPPF",
    "P5Transformer",
    "MobileNetV3BackboneDW",
    "UltraLiteNeckDW",
)


class DWConvCustom(nn.Module):
    """Depthwise Separable Conv: depthwise 3x3 + pointwise 1x1."""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, norm=True, activation=True):
        """Initialize depthwise separable convolution.
        
        Args:
            in_ch (int): Input channels
            out_ch (int): Output channels
            kernel_size (int): Kernel size for depthwise conv
            stride (int): Stride
            padding (int): Padding
            bias (bool): Use bias
            norm (bool): Use batch normalization
            activation (bool): Use activation
        """
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        self.use_norm = norm
        if norm:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = nn.Identity()
        self.use_act = activation
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        """Forward pass through depthwise separable convolution."""
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBNAct(nn.Module):
    """Standard conv 1x1 or 3x3 with BatchNorm and activation."""
    
    def __init__(self, in_ch, out_ch, k=1, stride=1, padding=0, act=True):
        """Initialize standard convolution block.
        
        Args:
            in_ch (int): Input channels
            out_ch (int): Output channels
            k (int): Kernel size
            stride (int): Stride
            padding (int): Padding
            act (bool): Use activation
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        """Forward pass through conv-bn-act."""
        return self.act(self.bn(self.conv(x)))


class CBAM_ChannelOnly(nn.Module):
    """Channel-only attention (lightweight CBAM variant)."""
    
    def __init__(self, channels, reduction=8):
        """Initialize channel attention module.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction ratio
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass with channel attention."""
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn


class SimSPPF(nn.Module):
    """Simplified SPPF (Spatial Pyramid Pooling - Fast)."""
    
    def __init__(self, c1, c2, k=5):
        """Initialize simplified SPPF module.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Kernel size for pooling
        """
        super().__init__()
        self.conv1 = ConvBNAct(c1, c2, k=1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)
        self.conv2 = ConvBNAct(c2 * 4, c2, k=1)
    
    def forward(self, x):
        """Forward pass through SPPF."""
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class P5Transformer(nn.Module):
    """Small transformer for P5 features."""
    
    def __init__(self, in_channels, embed_dim=48, ff_dim=96, num_layers=2):
        """Initialize transformer module for P5 features.
        
        Args:
            in_channels (int): Input channels
            embed_dim (int): Embedding dimension
            ff_dim (int): Feedforward dimension
            num_layers (int): Number of transformer layers
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=1,
                dim_feedforward=ff_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
    
    def forward(self, x):
        """Forward pass through transformer."""
        b, c, h, w = x.shape
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        for layer in self.layers:
            tokens = layer(tokens)
        x = tokens.transpose(1, 2).view(b, -1, h, w)
        return self.out_proj(x)


class MobileNetV3BackboneDW(nn.Module):
    """MobileNetV3 Small backbone with depthwise convolutions."""
    
    def __init__(self, pretrained=True):
        """Initialize MobileNetV3 backbone.
        
        Args:
            pretrained (bool): Use pretrained weights
        """
        super().__init__()
        m = mobilenet_v3_small(pretrained=pretrained)
        feats = m.features

        # Stage slicing for multi-scale features
        # feats[:3]  -> P3 (stride 8),  out_channels = 24
        # feats[3:7] -> P4 (stride 16), out_channels = 40
        # feats[7:]  -> P5 (stride 32), out_channels = 576
        self.stage1 = feats[:3]
        self.stage2 = feats[3:7]
        self.stage3 = feats[7:]

        # Depthwise convolutions for channel processing
        self.conv_p4_dw = DWConvCustom(40, 40, kernel_size=3, stride=1, padding=1)
        self.conv_p5_dw = DWConvCustom(576, 160, kernel_size=3, stride=1, padding=1)

        # Output channels for each stage
        self.out_channels = [24, 40, 160]

    def forward(self, x):
        """Forward pass through backbone.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: Multi-scale features [P3, P4, P5]
        """
        p3 = self.stage1(x)
        p4_in = self.stage2(p3)
        p4 = self.conv_p4_dw(p4_in)
        p5_in = self.stage3(p4)
        p5 = self.conv_p5_dw(p5_in)
        return [p3, p4, p5]


class UltraLiteNeckDW(nn.Module):
    """Ultra-lightweight neck with attention and transformer."""
    
    def __init__(self, in_channels=[24, 40, 160]):
        """Initialize ultra-lite neck.
        
        Args:
            in_channels (list): Input channels for [P3, P4, P5]
        """
        super().__init__()
        c3, c4, c5 = in_channels

        # Channel attention for each level
        self.cbam3 = CBAM_ChannelOnly(c3, reduction=8)
        self.cbam4 = CBAM_ChannelOnly(c4, reduction=8)

        # SPPF on P5
        self.sppf = SimSPPF(c5, c5)

        # Transformer on P5
        self.trans_p5 = P5Transformer(in_channels=c5, embed_dim=48, ff_dim=96, num_layers=2)
        self.cbam5 = CBAM_ChannelOnly(48, reduction=8)

        # Channel unification for outputs
        self.out3 = DWConvCustom(c3, 32, kernel_size=3, padding=1)
        self.out4 = DWConvCustom(c4, 48, kernel_size=3, padding=1)
        self.out5 = DWConvCustom(48, 64, kernel_size=3, padding=1)

        # Fusion layers
        self.fuse_p4 = ConvBNAct(48 + 64, 48, k=1)
        self.fuse_p3 = ConvBNAct(32, 32, k=1)

    def forward(self, feats):
        """Forward pass through neck.
        
        Args:
            feats (list): Multi-scale features [P3, P4, P5]
            
        Returns:
            list: Fused features [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = feats

        # Apply channel attention
        p3 = self.cbam3(p3)
        p4 = self.cbam4(p4)

        # P5 pipeline
        p5 = self.sppf(p5)
        p5 = self.trans_p5(p5)
        p5 = self.cbam5(p5)

        # Unify channels
        p3_u = self.out3(p3)
        p4_u = self.out4(p4)
        p5_u = self.out5(p5)

        # Fusion with pooling
        p5_pool_to_p4 = nn.functional.adaptive_avg_pool2d(p5_u, p4_u.shape[-2:])
        p4_u = self.fuse_p4(torch.cat([p4_u, p5_pool_to_p4], dim=1))
        p3_u = self.fuse_p3(p3_u)

        return [p3_u, p4_u, p5_u]
