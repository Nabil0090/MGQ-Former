import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, mask):
        x = mask.unsqueeze(1)
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        residual = query
        out, _ = self.attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        return self.norm(residual + self.drop(out))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out, _ = self.attn(x, x, x)
        return self.norm(residual + self.drop(out))


class FFN(nn.Module):
    def __init__(self, embed_dim=768, ffn_dim=3072, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class QFormerLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, ffn_dim=3072, dropout=0.1):
        super().__init__()
        self.self_attn   = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.cross_image = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.cross_text  = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.cross_mask  = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.ffn         = FFN(embed_dim, ffn_dim, dropout)

    def forward(self, queries, img_patches, token_embed,
                mask_tokens, text_attn_mask=None):
        text_pad_mask = None
        if text_attn_mask is not None:
            text_pad_mask = ~text_attn_mask

        queries = self.self_attn(queries)
        queries = self.cross_image(queries, img_patches)
        queries = self.cross_text(queries, token_embed, text_pad_mask)
        queries = self.cross_mask(queries, mask_tokens)
        queries = self.ffn(queries)
        return queries


class QFormer(nn.Module):
    def __init__(
        self,
        num_queries = 48,
        embed_dim   = 768,
        num_heads   = 8,
        num_layers  = 3,
        ffn_dim     = 3072,
        dropout     = 0.1,
    ):
        super().__init__()
        self.learned_queries = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            QFormerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_patches, cls_embed, token_embed,
                mask_tokens, text_attn_mask=None):
        B = img_patches.shape[0]
        queries  = self.learned_queries.expand(B, -1, -1).clone()
        cls_bias = self.cls_proj(cls_embed).unsqueeze(1)
        queries  = queries + cls_bias

        for layer in self.layers:
            queries = layer(
                queries, img_patches, token_embed,
                mask_tokens, text_attn_mask
            )

        return self.norm(queries)


class DualPathPool(nn.Module):
    def __init__(self, embed_dim=768, num_types=6):
        super().__init__()
        # global path — broad soft attention over all queries
        # captures scene-level composition (Comprehensive Analysis)
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.global_attn  = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.global_norm  = nn.LayerNorm(embed_dim)

        # local path — sharpened attention for object-level detail
        # captures fine-grained object state (Object Situation Analysis)
        self.local_query  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.local_attn   = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.local_norm   = nn.LayerNorm(embed_dim)

        # type-conditioned gate: predicts scalar in [0,1]
        # 0 = prefer global path, 1 = prefer local path
        self.gate = nn.Sequential(
            nn.Linear(num_types, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.fuse_norm = nn.LayerNorm(embed_dim)

    def forward(self, queries, type_logits_detached):
        B = queries.shape[0]

        # global path — standard soft attention
        gq = self.global_query.expand(B, -1, -1)
        g_out, _ = self.global_attn(gq, queries, queries)
        g_out = self.global_norm(g_out.squeeze(1))           # (B, 768)

        # local path — scale keys/values to sharpen attention distribution
        lq = self.local_query.expand(B, -1, -1)
        l_out, _ = self.local_attn(lq, queries * 1.5, queries * 1.5)
        l_out = self.local_norm(l_out.squeeze(1))            # (B, 768)

        # gate conditioned on lightweight type prediction
        gate  = self.gate(type_logits_detached)              # (B, 1)

        # weighted fusion
        pooled = (1 - gate) * g_out + gate * l_out          # (B, 768)
        return self.fuse_norm(pooled)


class AnswerHead(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=512, num_classes=147):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pooled):
        return self.net(pooled)


class TypeHead(nn.Module):
    def __init__(self, embed_dim=768, num_types=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_types),
        )

    def forward(self, pooled):
        return self.net(pooled)


class MaskedClassifier(nn.Module):
    def __init__(
        self,
        num_classes = 147,
        num_types   = 6,
        num_queries = 48,
        embed_dim   = 768,
        num_heads   = 8,
        num_layers  = 3,
        ffn_dim     = 3072,
        dropout     = 0.1,
    ):
        super().__init__()
        self.mask_encoder = MaskEncoder(out_dim=embed_dim)
        self.qformer      = QFormer(
            num_queries, embed_dim, num_heads,
            num_layers, ffn_dim, dropout
        )
        # lightweight type pre-classifier to guide pooling gate
        # uses mean of queries — cheap, no extra attention
        self.type_pre     = nn.Linear(embed_dim, num_types)
        self.pool         = DualPathPool(embed_dim, num_types)
        self.answer_head  = AnswerHead(embed_dim, 512, num_classes)
        self.type_head    = TypeHead(embed_dim, num_types)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(self, img_patches, cls_embed, token_embed,
                mask, attn_mask=None):
        mask_tokens = self.mask_encoder(mask)

        queries = self.qformer(
            img_patches, cls_embed, token_embed,
            mask_tokens, attn_mask
        )                                                     # (B, 48, 768)

        # lightweight type signal — detached so gate gradient
        # does not interfere with Q-Former training signal
        type_signal = self.type_pre(
            queries.mean(dim=1)
        ).detach()                                            # (B, 6)

        pooled = self.pool(queries, type_signal)             # (B, 768)

        ans_logits  = self.answer_head(pooled)
        type_logits = self.type_head(pooled)

        return ans_logits, type_logits


if __name__ == "__main__":
    B = 4
    model = MaskedClassifier(num_classes=147, num_types=6,
                             num_queries=48, num_layers=3)

    img_patches = torch.randn(B, 196, 768)
    cls_embed   = torch.randn(B, 768)
    token_embed = torch.randn(B, 14, 768)
    mask        = torch.rand(B, 224, 224)
    attn_mask   = torch.ones(B, 14, dtype=torch.bool)

    ans_logits, type_logits = model(
        img_patches, cls_embed, token_embed, mask, attn_mask
    )

    print(f"ans_logits  shape: {ans_logits.shape}")
    print(f"type_logits shape: {type_logits.shape}")

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\nTotal params:     {total:.1f}M")
    print(f"Trainable params: {trainable:.1f}M")
    print("\nSanity check passed. Ready for training.")
