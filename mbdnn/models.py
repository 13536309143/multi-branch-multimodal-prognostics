import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Module):
    """Explicit asymmetric SAME padding for an even 4x4 kernel."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4)

    def forward(self, x):
        return self.conv(F.pad(x, (1, 2, 1, 2)))


class SamePool(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        # ReLU precedes pooling, so zero padding matches paper semantics.
        return F.max_pool2d(F.pad(x, (0, (-w) % 4, 0, (-h) % 4)), 4, 4)


class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(SameConv(1, 4), nn.ReLU(), SamePool(),
                                      SameConv(4, 4), nn.ReLU(), SamePool(),
                                      SameConv(4, 8), nn.ReLU(), SamePool())
        # 325x194 -> 82x49 -> 21x13 -> 6x4, 8 channels.
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(8 * 6 * 4, 4), nn.Sigmoid())

    def forward(self, x):
        return self.dense(self.features(x))


class TextBranch(nn.Module):
    def __init__(self, embeddings, trainable=False, dense_activation="sigmoid"):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embeddings, dtype=torch.float32), freeze=not trainable, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(100, 32, window) for window in (2, 3, 4))
        if dense_activation not in ("sigmoid", "relu"):
            raise ValueError("Unknown text dense activation")
        self.dense = nn.Sequential(nn.Linear(96, 4), nn.Sigmoid() if dense_activation == "sigmoid" else nn.ReLU())

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = [F.relu(conv(embedded)).amax(dim=-1) for conv in self.convs]
        return self.dense(torch.cat(pooled, dim=1))


class MultiBranchDNN(nn.Module):
    def __init__(self, modalities="itn", embeddings=None, embedding_trainable=False, text_dense_activation="sigmoid"):
        super().__init__()
        if not modalities or any(c not in "itn" for c in modalities) or len(set(modalities)) != len(modalities):
            raise ValueError("Invalid modalities")
        self.modalities = "".join(c for c in "itn" if c in modalities)
        self.branches = nn.ModuleDict()
        if "i" in modalities:
            self.branches["i"] = ImageBranch()
        if "t" in modalities:
            if embeddings is None:
                raise ValueError("Text branch requires embeddings")
            self.branches["t"] = TextBranch(embeddings, embedding_trainable, text_dense_activation)
        if "n" in modalities:
            self.branches["n"] = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
        self.regression = nn.Linear(4 * len(modalities), 1)
        # Initialization is an explicit reconstruction choice.
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, batch):
        keys = {"i": "image", "t": "text", "n": "numeric"}
        features = [self.branches[c](batch[keys[c]]) for c in self.modalities]
        return self.regression(torch.cat(features, dim=1)).squeeze(-1)
