import torch
from vit_pytorch.vit import ViT
from vit_pytorch.utils import Config


def test():
    config = Config(
        num_heads=8,
        emb_size=768,
        channel=3,
        dropout=0.1,
        pre_LM=False,
        hidden_dim=3072,
        depth=6,
        img_size=(224, 224),
        patch_size=(8, 8),
        emb_dropout=0.0,
        num_classes=100,
    )
    model = ViT(config)
    image = torch.randn(1, 3, 224, 224)
    y_hat = model(image)
    assert y_hat.shape == (1, config.num_classes), "Incorrect logits output"


if __name__ == "__main__":
    test()
