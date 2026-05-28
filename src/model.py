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
            embedding_dim=32
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, 32)
        )

        self.positional_embedding = nn.Embedding(
            51,
            32
        )

        self.blocks = nn.ModuleList([

            FlashAttentionBlock(
                embed_dim=32,
                num_heads=4,
                dropout=0.1
            )

            for _ in range(4)

        ])

        self.layernorm = nn.LayerNorm(
            32
        )

        self.shared_mlp = nn.Sequential(

            nn.Linear(32, 16),

            nn.GELU(),

            nn.Dropout(0.2)

        )

        self.atac_head = nn.Linear(
            16,
            1
        )

        self.h3k27ac_head = nn.Linear(
            16,
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

class InceptionResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        branch_channels = out_channels // 4

        # (3x3conv)

        self.branch3 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=1
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU(),

            nn.Conv1d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU()

        )

        # (5x5 conv)

        self.branch5 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=1
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU(),

            nn.Conv1d(
                branch_channels,
                branch_channels,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU()

        )

        # 9x9 conv

        self.branch9 = nn.Sequential(

            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=1
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU(),

            nn.Conv1d(
                branch_channels,
                branch_channels,
                kernel_size=9,
                padding=4
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU()

        )

        # Pooling branch

        self.pool_branch = nn.Sequential(

            nn.MaxPool1d(
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=1
            ),

            nn.BatchNorm1d(
                branch_channels
            ),

            nn.GELU()

        )

        self.projection = nn.Sequential(

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=1
            ),

            nn.BatchNorm1d(
                out_channels
            )

        )

        if in_channels != out_channels:

            self.residual_projection = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.residual_projection = nn.Identity()

        self.final_activation = nn.GELU()

    def forward(
        self,
        x
    ):

        residual = self.residual_projection(
            x
        )

        branch3 = self.branch3(x)

        branch5 = self.branch5(x)

        branch9 = self.branch9(x)

        pool_branch = self.pool_branch(x)

        out = torch.cat(

            [
                branch3,
                branch5,
                branch9,
                pool_branch
            ],

            dim=1

        )

        out = self.projection(
            out
        )

        out = out + residual

        out = self.final_activation(
            out
        )

        return out

class DilatedResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        dilation=2,
        dropout=0.1
    ):

        super().__init__()

        padding = dilation

        self.block = nn.Sequential(

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation
            ),

            nn.BatchNorm1d(
                channels
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation
            ),

            nn.BatchNorm1d(
                channels
            )

        )

        self.final_activation = nn.GELU()

    def forward(
        self,
        x
    ):

        residual = x

        out = self.block(
            x
        )

        out = out + residual

        out = self.final_activation(
            out
        )

        return out

class RegulatoryResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.GELU()

        )

        self.inception_blocks = nn.Sequential(

            InceptionResidualBlock(
                in_channels=64,
                out_channels=64
            ),

            InceptionResidualBlock(
                in_channels=64,
                out_channels=64
            ),

            InceptionResidualBlock(
                in_channels=64,
                out_channels=64
            )

        )

        self.dilated_blocks = nn.Sequential(

            DilatedResidualBlock(
                channels=64,
                dilation=2
            ),

            DilatedResidualBlock(
                channels=64,
                dilation=4
            ),

            DilatedResidualBlock(
                channels=64,
                dilation=8
            )

        )

        self.global_pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.shared_mlp = nn.Sequential(

            nn.Linear(
                64,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.2
            )

        )

        self.atac_head = nn.Linear(
            128,
            1
        )

        self.h3k27ac_head = nn.Linear(
            128,
            1
        )

    def forward(
        self,
        x
    ):

        x = self.stem(
            x
        )

        x = self.inception_blocks(
            x
        )

        x = self.dilated_blocks(
            x
        )

        x = self.global_pool(
            x
        )

        x = x.squeeze(-1)

        shared_embedding = self.shared_mlp(
            x
        )


        atac_logits = self.atac_head(
            shared_embedding
        )

        h3k27ac_output = self.h3k27ac_head(
            shared_embedding
        )

        return (

            atac_logits,

            h3k27ac_output,

            shared_embedding

        )