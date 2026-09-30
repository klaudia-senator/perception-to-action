"""Compact multi-head U-Net used for optional PyTorch inference.

No trained weights are distributed with this repository. The demo defaults to
an explainable synthetic backend so the full pipeline can run immediately.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without the ML extra
    torch = None
    nn = None


if nn is not None:

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class MultiHeadUNet(nn.Module):
        """Small U-Net with a shared encoder-decoder and one logit head per class."""

        def __init__(
            self,
            in_channels: int = 3,
            heads: tuple[str, ...] = ("foreground", "target", "context", "marker"),
            base_channels: int = 16,
        ) -> None:
            super().__init__()
            self.head_names = heads
            self.enc1 = DoubleConv(in_channels, base_channels)
            self.enc2 = DoubleConv(base_channels, base_channels * 2)
            self.bridge = DoubleConv(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
            self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
            self.dec1 = DoubleConv(base_channels * 2, base_channels)
            self.heads = nn.ModuleDict(
                {name: nn.Conv2d(base_channels, 1, kernel_size=1) for name in heads}
            )

        def forward(self, x):
            skip1 = self.enc1(x)
            skip2 = self.enc2(self.pool(skip1))
            latent = self.bridge(self.pool(skip2))
            decoded2 = self.dec2(torch.cat((self.up2(latent), skip2), dim=1))
            decoded1 = self.dec1(torch.cat((self.up1(decoded2), skip1), dim=1))
            return {name: layer(decoded1) for name, layer in self.heads.items()}


else:

    class MultiHeadUNet:  # pragma: no cover - simple dependency guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is optional. Install it with: pip install -r requirements-ml.txt"
            )


class TorchInferenceBackend:
    """Turns an RGB frame into named probability maps using a PyTorch model."""

    def __init__(self, model, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("PyTorch is required for TorchInferenceBackend")
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, frame):
        import numpy as np

        image = np.asarray(frame, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        return {
            name: torch.sigmoid(value)[0, 0].detach().cpu().numpy()
            for name, value in logits.items()
        }

