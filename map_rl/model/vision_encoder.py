import torch
import torch.nn as nn
from torchvision import transforms

WEIIGHT_PATH = 'https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNnp2ZGwxMGU4bmNvZXA2ZTBnNzQ5aDV3IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU4NDE1OTR9fX1dfQ__&Signature=IeOOXtAhkZykP5ZkZdUIA8fnu-RiaQmuKWi7UO5tNvRWuL41Gg0bYB3vNFoQsUG%7EytxhhUvY2IpR7gXiyAOrlt-nPaYlg46oBCgONrJaIGASZWvSE3bf-UlTpmEKUVJaV4JSVDLLJUqnjSm6Q1pOkZKJKs7xUrQt6-GxalOh81OeaYNeKAyjl0R78LUOfnpawz6OSvJLWDHNgbiyieLQOAVTj7Q-%7EwqFG8K03qlNnA2xR5twxTIz1H8jrL7tAncheODNFmmwWSWgE495O4RsD09S7LyuhxxqTnXfOmwkfUf1OZjSIDHsvzrqp1Ap31sLW5mYmrf2Ruf7IA3akO4YmA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1290416756057877'

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
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load backbone lazily via torch.hub to avoid extra dependencies
        # self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone = torch.hub.load('dinov3', 'dinov3_vits16', source='local', weights=WEIIGHT_PATH)
        self.dino_output_dim = 384
        self.dino_proj = nn.Conv2d(self.dino_output_dim, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # DINOv2 normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    def _forward_dino_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x)

        for blk in self.backbone.blocks:
            if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                raise ValueError("Rope embedding not found in DINOv3")
            x = blk(x, rope_sincos)
        x = self.backbone.norm(x)  # (B, 1 + N, C)
        x = x[:, 5:, :]  # drop CLS and storage tokens

        return x

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = embed_dim and Hf = H//16, Wf = W//16
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        # Normalize per DINOv2 recipe
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        tokens = self._forward_dino_tokens(images_bchw)  # (B, N, C)

        C = self.dino_output_dim
        Hf, Wf = H // 16, W // 16
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()
        fmap = self.dino_proj(fmap)
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