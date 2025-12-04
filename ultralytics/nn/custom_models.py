# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom MobileNetV3-YOLO model integration."""

import torch
import torch.nn as nn

from ultralytics.nn.modules import MobileNetV3BackboneDW, UltraLiteNeckDW, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER


class MobileNetV3YOLO(nn.Module):
    """Custom YOLO model with MobileNetV3 backbone and ultra-lite neck.
    
    This model combines:
    - MobileNetV3 Small backbone with depthwise convolutions
    - Ultra-lightweight neck with attention and transformer
    - Standard YOLOv8 detection head
    
    Attributes:
        backbone (MobileNetV3BackboneDW): Feature extraction backbone
        neck (UltraLiteNeckDW): Feature fusion neck
        head (Detect): Detection head
        stride (torch.Tensor): Model stride values
        names (dict): Class names
    """
    
    def __init__(self, nc=80, pretrained=True, verbose=True):
        """Initialize MobileNetV3-YOLO model.
        
        Args:
            nc (int): Number of classes
            pretrained (bool): Use pretrained MobileNetV3 backbone
            verbose (bool): Print model information
        """
        super().__init__()
        self.nc = nc
        
        # Initialize backbone and neck
        self.backbone = MobileNetV3BackboneDW(pretrained=pretrained)
        self.neck = UltraLiteNeckDW(in_channels=[24, 40, 160])
        
        # Neck output channels: [32, 48, 64] for [P3, P4, P5]
        neck_out_channels = [32, 48, 64]
        
        # Initialize detection head
        # YOLOv8n-style head with our custom neck outputs
        self.head = Detect(nc=nc, ch=tuple(neck_out_channels))
        
        # Model metadata
        self.names = {i: f"{i}" for i in range(nc)}
        self.stride = torch.tensor([8, 16, 32])  # P3, P4, P5 strides
        
        # Initialize head
        self._initialize_head()
        
        if verbose:
            self.info()
    
    def _initialize_head(self):
        """Initialize detection head with proper strides."""
        m = self.head
        if isinstance(m, Detect):
            s = 640  # default image size
            m.inplace = True
            
            # Forward pass to get strides
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])
            self.stride = m.stride
            m.bias_init()  # Initialize biases
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
            
        Returns:
            (torch.Tensor): Detection outputs
        """
        # Backbone: extract multi-scale features
        feats = self.backbone(x)  # [P3, P4, P5] with channels [24, 40, 160]
        
        # Neck: fuse features
        fused_feats = self.neck(feats)  # [P3, P4, P5] with channels [32, 48, 64]
        
        # Head: generate predictions
        outputs = self.head(fused_feats)
        
        return outputs
    
    def fuse(self, verbose=True):
        """Fuse Conv2d + BatchNorm2d layers for inference optimization.
        
        Args:
            verbose (bool): Print fusion information
            
        Returns:
            (MobileNetV3YOLO): Fused model
        """
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)) and hasattr(m, 'bn'):
                    # Fuse convolution and batch norm
                    from ultralytics.utils.torch_utils import fuse_conv_and_bn
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
            if verbose:
                self.info()
        return self
    
    def is_fused(self, thresh=10):
        """Check if model has less than threshold BatchNorm layers.
        
        Args:
            thresh (int): Threshold for BatchNorm layers
            
        Returns:
            (bool): True if fused
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh
    
    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.
        
        Args:
            detailed (bool): Show detailed layer information
            verbose (bool): Print to console
            imgsz (int): Input image size for computation
        """
        from ultralytics.utils.torch_utils import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    
    def predict(self, x, profile=False, visualize=False, augment=False):
        """Perform inference.
        
        Args:
            x (torch.Tensor): Input tensor
            profile (bool): Profile computation time
            visualize (bool): Visualize features
            augment (bool): Apply test-time augmentation
            
        Returns:
            (torch.Tensor): Model predictions
        """
        if augment:
            # Simple TTA: original + horizontal flip
            y = []
            for xi in [x, torch.flip(x, [-1])]:
                yi = self.forward(xi)
                y.append(yi)
            return torch.cat(y, -1)
        return self.forward(x)
    
    @property
    def task(self):
        """Return task type."""
        return 'detect'
