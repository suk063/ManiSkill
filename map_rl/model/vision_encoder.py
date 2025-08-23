import torch
import torch.nn as nn
from torchvision import transforms

class DINO2DFeatureEncoder(nn.Module):
    """
    Thin wrapper around DINOv2 ViT-S/14 to produce dense 2D feature maps.

    Inputs are expected in shape (B, C, H, W) with values in [0, 1].
    Normalization is applied internally.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        model_name: str = "dinov2_vits14",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Load backbone lazily via torch.hub to avoid extra dependencies
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        # self.backbone = torch.hub.load('dinov3', 'dinov3_vits16', source='local', weights=WEIIGHT_PATH)
        self.dino_output_dim = 384
        self.dino_head = nn.Sequential(
            nn.Conv2d(self.dino_output_dim, 256, kernel_size=1, bias=False),
            nn.GroupNorm(1, 256),  # Equivalent to LayerNorm for channels
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        )
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _forward_dino_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)  # (B, 1 + N, C)
        x = x[:, 1:, :]            # drop CLS → (B, N, C)
        return x

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = embed_dim and Hf = H//14, Wf = W//14
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()

        B, _, H, W = images_bchw.shape
        
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self._forward_dino_tokens(images_bchw)
        else:
            tokens = self._forward_dino_tokens(images_bchw)

        C = self.dino_output_dim
        Hf, Wf = H // 14, W // 14
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()
        fmap = self.dino_head(fmap)
        
        return fmap

class PlainCNNFeatureEncoder(nn.Module):
    """
    Lightweight CNN that produces a dense 2D feature map.

    - Inputs are expected in shape (B, C, H, W) with values in [0, 1].
    - Uses only convolution layers (no pooling) to downsample.
      For 84×84 inputs, the output spatial size is 6×6.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 84 -> 42 -> 21 -> 11 -> 6 using stride-2, k=3, p=1 convs
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        x = self.cnn(images_bchw)

        return x