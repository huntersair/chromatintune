import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceCNN(nn.Module):

    def __init__(
        self,
        use_h3k27ac=True
    ):

        super().__init__()

        self.use_h3k27ac = use_h3k27ac

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=11
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        if self.use_h3k27ac:

            self.fc = nn.Linear(
                65,
                1
            )

        else:

            self.fc = nn.Linear(
                64,
                1
            )

    def forward(
        self,
        sequence,
        h3k27ac=None
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        if self.use_h3k27ac:

            x = torch.cat(
                [x, h3k27ac],
                dim=1
            )

        x = self.fc(x)

        return x

class DeepSequenceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=11,
            padding=5
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            padding=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(
            p=0.3
        )

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.fc2 = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        sequence
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

class SequenceTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=64
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=200,
            embedding_dim=64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        tokens
    ):

        positions = torch.arange(
            0,
            tokens.size(1),
            device=tokens.device
        ).unsqueeze(0)

        x = self.embedding(tokens)

        x = x + self.positional_embedding(positions)

        x = self.transformer(x)

        x = x.transpose(1, 2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x

class HybridCNNTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=64,
            kernel_size=11,
            padding=5
        )

        self.relu = nn.ReLU()

        self.position_embedding = nn.Embedding(
            200,
            64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            64,
            32
        )

        self.fc2 = nn.Linear(
            32,
            1
        )

    def forward(
        self,
        sequence
    ):

        x = self.conv1(sequence)

        x = self.relu(x)

        # converting to (batch, seq_len, channels)

        x = x.transpose(1, 2)

        positions = torch.arange(
            0,
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        x = x + self.position_embedding(
            positions
        )

        x = self.transformer(x)

        x = x.transpose(1, 2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x

class FlashTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=128
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, 128)
        )

        self.positional_embedding = nn.Embedding(
            201,
            128
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.layernorm = nn.LayerNorm(
            128
        )

        self.dropout = nn.Dropout(
            0.2
        )

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.fc2 = nn.Linear(
            64,
            1
        )

        self.gelu = nn.GELU()

    def forward(
        self,
        tokens
    ):

        batch_size = tokens.size(0)

        x = self.embedding(tokens)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        positions = torch.arange(
            0,
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        x = x + self.positional_embedding(
            positions
        )

        x = self.transformer(x)

        cls_output = x[:, 0]

        cls_output = self.layernorm(
            cls_output
        )

        cls_output = self.dropout(
            cls_output
        )

        cls_output = self.fc1(
            cls_output
        )

        cls_output = self.gelu(
            cls_output
        )

        cls_output = self.fc2(
            cls_output
        )

        return cls_output

class FlashAttentionBlock(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        dropout=0.1
    ):

        super().__init__()

        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim

        self.head_dim = embed_dim // num_heads

        self.attention_dropout = dropout

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.layernorm1 = nn.LayerNorm(
            embed_dim
        )

        self.layernorm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.residual_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):

        residual = x

        x = self.layernorm1(x)

        B, N, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        attention_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.attention_dropout,
            is_causal=False
        )

        attention_output = attention_output.transpose(
            1,
            2
        ).contiguous()

        attention_output = attention_output.view(
            B,
            N,
            C
        )

        attention_output = self.out_proj(
            attention_output
        )

        x = residual + self.residual_dropout(
            attention_output
        )

        residual = x

        x = self.layernorm2(x)

        x = residual + self.mlp(x)

        return x

class LongContextFlashAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=128
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, 128)
        )

        self.positional_embedding = nn.Embedding(
            1001,
            128
        )

        self.blocks = nn.ModuleList([

            FlashAttentionBlock(
                embed_dim=128,
                num_heads=8,
                dropout=0.1
            )

            for _ in range(4)

        ])

        self.layernorm = nn.LayerNorm(
            128
        )

        self.shared_mlp = nn.Sequential(

            nn.Linear(128, 64),

            nn.GELU(),

            nn.Dropout(0.2)

        )

        self.atac_head = nn.Linear(
            64,
            1
        )

        self.h3k27ac_head = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        tokens
    ):

        batch_size = tokens.size(0)

        x = self.embedding(tokens)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_tokens, x],
            dim=1
        )

        positions = torch.arange(
            0,
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        x = x + self.positional_embedding(
            positions
        )

        for block in self.blocks:

            x = block(x)

        raw_cls_embedding = x[:, 0]

        raw_cls_embedding = self.layernorm(
            raw_cls_embedding
        )

        shared_features = self.shared_mlp(
            raw_cls_embedding
        )

        atac_logits = self.atac_head(
            shared_features
        )

        h3k27ac_output = self.h3k27ac_head(
            shared_features
        )

        return (
            atac_logits,
            h3k27ac_output,
            raw_cls_embedding
        )
