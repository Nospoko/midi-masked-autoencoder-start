import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from models.modules.masking import random_masking
from models.modules.positional_embedding import SinusoidalPositionEmbeddings


class OutputBlock(nn.Module):
    def __init__(self, features: int, out_features: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class DynamicsEmbeddingLayer(nn.Module):
    def __init__(self, in_features: int = 3, hidden_features: int = 384, out_features: int = 384, depth: int = 4):
        super().__init__()

        layer_dims = [in_features] + [hidden_features for _ in range(depth)]

        layers = []

        for ins, outs in zip(layer_dims[:-1], layer_dims[1:]):
            layers += [nn.Linear(ins, outs), nn.SiLU()]

        layers += [nn.Linear(layer_dims[-1], out_features)]

        self.dynamics_embedding = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.dynamics_embedding(x)


class MidiMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dynamics_embedding_depth: int = 4,
    ):
        super().__init__()

        # embedding
        encoder_quarter_dim = encoder_dim // 4
        self.pitch_embedding = nn.Embedding(num_embeddings=88, embedding_dim=encoder_quarter_dim)
        # self.dynamics_embedding = nn.Linear(3, encoder_half_dim)
        self.dynamics_embedding = DynamicsEmbeddingLayer(
            in_features=3,
            hidden_features=3 * encoder_quarter_dim,
            out_features=3 * encoder_quarter_dim,
            depth=dynamics_embedding_depth,
        )

        # encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        self.encoder_pos_emb = SinusoidalPositionEmbeddings(encoder_dim)

        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=encoder_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_out_norm = nn.LayerNorm(encoder_dim)

        # decoder
        self.decoder_embedding = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_emb = SinusoidalPositionEmbeddings(decoder_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_out_norm = nn.LayerNorm(decoder_dim)

        self.pitch_out = OutputBlock(decoder_dim, out_features=88)
        self.velocity_out = OutputBlock(decoder_dim, out_features=1)
        self.start_out = OutputBlock(decoder_dim, out_features=1)
        self.duration_out = OutputBlock(decoder_dim, out_features=1)

    def forward_encoder(
        self, pitch: torch.Tensor, velocity: torch.Tensor, start: torch.Tensor, duration: torch.Tensor, masking_ratio: float
    ):
        N, L = pitch.shape

        pitch_emb = self.pitch_embedding(pitch)
        dynamics = torch.stack([velocity, start, duration], dim=-1)
        dynamics_emb = self.dynamics_embedding(dynamics)

        # shape: [batch_size, seq_len, embedding_dim]
        x = torch.cat([pitch_emb, dynamics_emb], dim=-1)

        # seq_len + cls_token
        positions = torch.arange(L + 1, device=x.device, dtype=torch.float32)
        # shape: [batch_size, seq_len, embedding_dim]
        pe = self.encoder_pos_emb(positions)[None, :, :]

        # adding positional information
        x = x + pe[:, 1:, :]

        # masking input
        x, mask, ids_restore = random_masking(x, masking_ratio=masking_ratio)

        # appending cls token
        cls_token = self.cls_token + pe[:, :1, :]
        cls_tokens = cls_token.expand(N, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # encoder
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_out_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embedding(x)

        # append masked tokens
        # ids_restore.shape[1] + 1 - x.shape[1] means: whole sequence len + cls_token - not masked tokens = masked tokens len
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # cat without cls token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # seq_len + cls_token
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        # shape: [batch_size, seq_len, embedding_dim]
        pe = self.decoder_pos_emb(positions)[None, :, :]

        # adding positional information
        x = x + pe

        # decoder
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_out_norm(x)
        x = x[:, 1:, :]

        pred_pitch = self.pitch_out(x)
        pred_velocity = self.velocity_out(x)[:, :, 0]
        pred_start = self.start_out(x)[:, :, 0]
        pred_duration = self.duration_out(x)[:, :, 0]

        return pred_pitch, pred_velocity, pred_start, pred_duration

    def forward(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        start: torch.Tensor,
        duration: torch.Tensor,
        masking_ratio: float = 0.15,
    ):
        latent, mask, ids_restore = self.forward_encoder(pitch, velocity, start, duration, masking_ratio=masking_ratio)
        pred_pitch, pred_velocity, pred_start, pred_duration = self.forward_decoder(latent, ids_restore)

        return pred_pitch, pred_velocity, pred_start, pred_duration, mask
