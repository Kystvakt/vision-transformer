from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


### The following are the experimental elements. ###

# Attention block
class Attention(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.num_heads = config.num_heads
        self.scale_factor = config.dim_head ** -0.5

        self.qkv = nn.Linear(config.emb_size, config.emb_size * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.project = nn.Linear(config.emb_size, config.emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), self.qkv(x).chunk(3, dim=-1))

        dot = torch.einsum('bhqd, bhkd -> bhqk', q, k)  # batch, num_heads, query_len, key_len
        attn = self.softmax(dot * self.scale_factor)
        attn = self.dropout(attn)

        output = torch.einsum('bhal, bhlv -> bhav', attn, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.project(output)
        output = self.dropout(output)
        return output


# New GELU
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# Residual
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# Layer Normalization
# class LayerNorm(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.norm = nn.LayerNorm(config.norm_dim)
#
#     def forward(self, x):
#         return self.norm(x)


# Fully-connected feed-forward network
# class FeedForward(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(config.emb_size, config.hidden_dim),
#             NewGELU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(config.hidden_dim, config.emb_size),
#             nn.Dropout(config.dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
class FeedForward(nn.Sequential):
    def __init__(self, config):
        super().__init__(
            nn.Linear(config.emb_size, config.hidden_dim),
            NewGELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emb_size),
            nn.Dropout(config.dropout)
        )

# Transformer block
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(config),
                    nn.LayerNorm(config.emb_size),
                    FeedForward(config),
                    LayerNorm(config.emb_size)
                ])
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_h, img_w = config.size if isinstance(config.size, tuple) else (config.size, config.size)
        p_h, p_w = config.size if isinstance(config.size, tuple) else (config.size, config.size)

        assert img_h % p_h == 0 and img_w % p_w == 0, "Image dimensions must be divisible by the patch size"

        self.patch_size = config.patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(config.channel, config.emb_size, kernel_size=config.patch_size, stride=config.patch_size),
            Rearrange('b c (h) (w) -> b (h w) c')
        )
        self.cls_tkn = nn.Parameter(torch.randn(1, 1), config.emb_size)
        self.pos_emb = nn.Parameter(torch.randn(1, (img_h // p_h) * (img_w // p_w) + 1, config.emb_size))
        self.dropout = nn.Dropout(config.emb_dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tkn = repeat(self.cls_tkn, '() n c -> b n c', b=b)
        x = torch.cat((cls_tkn, x), dim=1)
        x += self.pos_emb
        return self.dropout(x)


# Classification head
# class ClassificationHead(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.net = nn.Sequential(
#             Reduce('b n c -> b c', reduction='mean'),
#             nn.LayerNorm(config.emb_size),
#             nn.Linear(config.emb_size, config.num_classes)
#         )
class ClassificationHead(nn.Sequential):
    def __init__(self, config):
        super().__init__(
            Reduce('b n c -> b c', reduction='mean'),
            nn.LayerNorm(config.emb_size),
            nn.Linear(config.emb_size, config.num_classes)
        )


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patchembedding = PatchEmbedding(config)
        self.transformer = Transformer(config)
        self.classificationhead = ClassificationHead(config)

    def forward(self, img):
        x = self.patchembedding(img)
        x = self.transformer(x)
        x = self.classificationhead(x)
        return x
