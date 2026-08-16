import math
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


QPOS_SLOTS = 5
QVEL_SLOTS = 6
QFRC_SLOTS = 6
CINERT_SLOTS = 10
CVEL_SLOTS = 6
CFRC_SLOTS = 6

QPOS_OFFSET = 0
QVEL_OFFSET = QPOS_OFFSET + QPOS_SLOTS
QFRC_OFFSET = QVEL_OFFSET + QVEL_SLOTS
CINERT_OFFSET = QFRC_OFFSET + QFRC_SLOTS
CVEL_OFFSET = CINERT_OFFSET + CINERT_SLOTS
CFRC_OFFSET = CVEL_OFFSET + CVEL_SLOTS
BODY_FEATURE_WIDTH = CFRC_OFFSET + CFRC_SLOTS


def _canonical_env_name(env_name: str) -> str:
    name = str(env_name).lower().replace("-", "").replace("_", "")
    for suffix in ("v2", "v3", "v4", "v5"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    aliases = {
        "halfcheetah": "halfcheetah",
        "hopper": "hopper",
        "walker2d": "walker2d",
        "swimmer": "swimmer",
        "ant": "ant",
        "humanoid": "humanoid",
        "humanoidstandup": "humanoidstandup",
    }
    if name not in aliases:
        raise ValueError(f"Unsupported MuJoCo Transformer environment: {env_name}")
    return aliases[name]


def _empty_layout(num_tokens: int) -> torch.Tensor:
    return torch.full((num_tokens, BODY_FEATURE_WIDTH), -1, dtype=torch.long)


def _put(layout: torch.Tensor, token: int, offset: int, indices: Sequence[int]) -> None:
    indices = list(indices)
    if not indices:
        return
    end = offset + len(indices)
    if end > layout.shape[1]:
        raise ValueError(f"Token feature assignment exceeds width: {offset}:{end}")
    if torch.any(layout[token, offset:end] >= 0):
        raise ValueError(f"Token feature slots already assigned: token={token}, slots={offset}:{end}")
    layout[token, offset:end] = torch.as_tensor(indices, dtype=torch.long)


def _simple_joint_layout(
    root_qpos: Sequence[int],
    root_qvel: Sequence[int],
    joint_qpos: Sequence[int],
    joint_qvel: Sequence[int],
) -> torch.Tensor:
    if len(joint_qpos) != len(joint_qvel):
        raise ValueError("Joint qpos/qvel lists must have equal length")
    layout = _empty_layout(1 + len(joint_qpos))
    _put(layout, 0, QPOS_OFFSET, root_qpos)
    _put(layout, 0, QVEL_OFFSET, root_qvel)
    for token, (qpos_index, qvel_index) in enumerate(zip(joint_qpos, joint_qvel), start=1):
        _put(layout, token, QPOS_OFFSET, [qpos_index])
        _put(layout, token, QVEL_OFFSET, [qvel_index])
    return layout


def _humanoid_layout() -> torch.Tensor:
    # Gymnasium Humanoid-v4 observation:
    # qpos[2:] (22), qvel (23), cinert (14*10), cvel (14*6),
    # qfrc_actuator (23), cfrc_ext (14*6).
    layout = _empty_layout(14)

    root_body = 1
    joint_bodies = [2, 2, 3, 4, 4, 4, 5, 7, 7, 7, 8, 10, 10, 11, 12, 12, 13]

    _put(layout, root_body, QPOS_OFFSET, range(0, 5))
    _put(layout, root_body, QVEL_OFFSET, range(22, 28))
    _put(layout, root_body, QFRC_OFFSET, range(269, 275))

    per_body_qpos: Dict[int, List[int]] = {body: [] for body in range(14)}
    per_body_qvel: Dict[int, List[int]] = {body: [] for body in range(14)}
    per_body_qfrc: Dict[int, List[int]] = {body: [] for body in range(14)}
    for coord, body in enumerate(joint_bodies):
        per_body_qpos[body].append(5 + coord)
        per_body_qvel[body].append(28 + coord)
        per_body_qfrc[body].append(275 + coord)

    for body in range(14):
        _put(layout, body, QPOS_OFFSET, per_body_qpos[body])
        _put(layout, body, QVEL_OFFSET, per_body_qvel[body])
        _put(layout, body, QFRC_OFFSET, per_body_qfrc[body])
        _put(layout, body, CINERT_OFFSET, range(45 + 10 * body, 45 + 10 * (body + 1)))
        _put(layout, body, CVEL_OFFSET, range(185 + 6 * body, 185 + 6 * (body + 1)))
        _put(layout, body, CFRC_OFFSET, range(292 + 6 * body, 292 + 6 * (body + 1)))

    return layout


def build_mujoco_body_layout(env_name: str, obs_dim: int) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    env_name = _canonical_env_name(env_name)
    if env_name == "ant":
        layout = _simple_joint_layout(
            root_qpos=range(0, 5),
            root_qvel=range(13, 19),
            joint_qpos=range(5, 13),
            joint_qvel=range(19, 27),
        )
        names = ("torso",) + tuple(f"joint_{index}" for index in range(8))
    elif env_name == "halfcheetah":
        layout = _simple_joint_layout(
            root_qpos=(0, 1),
            root_qvel=(8, 9, 10),
            joint_qpos=range(2, 8),
            joint_qvel=range(11, 17),
        )
        names = ("torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
    elif env_name == "hopper":
        layout = _simple_joint_layout(
            root_qpos=(0, 1),
            root_qvel=(5, 6, 7),
            joint_qpos=range(2, 5),
            joint_qvel=range(8, 11),
        )
        names = ("torso", "thigh", "leg", "foot")
    elif env_name == "walker2d":
        layout = _simple_joint_layout(
            root_qpos=(0, 1),
            root_qvel=(8, 9, 10),
            joint_qpos=range(2, 8),
            joint_qvel=range(11, 17),
        )
        names = ("torso", "thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left")
    elif env_name == "swimmer":
        layout = _simple_joint_layout(
            root_qpos=(0,),
            root_qvel=(3, 4, 5),
            joint_qpos=(1, 2),
            joint_qvel=(6, 7),
        )
        names = ("torso", "mid", "back")
    elif env_name in {"humanoid", "humanoidstandup"}:
        layout = _humanoid_layout()
        names = (
            "world",
            "torso",
            "lwaist",
            "pelvis",
            "right_thigh",
            "right_shin",
            "right_foot",
            "left_thigh",
            "left_shin",
            "left_foot",
            "right_upper_arm",
            "right_lower_arm",
            "left_upper_arm",
            "left_lower_arm",
        )
    else:
        raise AssertionError("canonical environment dispatch is incomplete")

    used = layout[layout >= 0]
    expected = torch.arange(obs_dim, dtype=torch.long)
    actual = torch.sort(used).values
    if actual.numel() != expected.numel() or not torch.equal(actual, expected):
        missing = sorted(set(expected.tolist()) - set(actual.tolist()))
        duplicate_count = int(actual.numel() - torch.unique(actual).numel())
        raise ValueError(
            f"Invalid {env_name} body-token layout for obs_dim={obs_dim}: "
            f"missing={missing}, duplicate_count={duplicate_count}, used={actual.numel()}"
        )
    if len(names) != layout.shape[0]:
        raise ValueError("Token name/layout count mismatch")
    return layout, names


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


def _token_linear(layer: nn.Linear, tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    output = layer(tensor.reshape(-1, shape[-1]))
    return output.reshape(*shape[:-1], output.shape[-1])


class SingleLayerBodyTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        env_name: str,
        d_model: int,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be positive")

        layout, token_names = build_mujoco_body_layout(env_name, obs_dim)
        self.obs_dim = int(obs_dim)
        self.env_name = _canonical_env_name(env_name)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.num_body_tokens = int(layout.shape[0])
        self.sequence_length = self.num_body_tokens + 1
        self.token_names = token_names
        self.kfac_weight_sharing = "expand"

        gather_mask = layout >= 0
        self.register_buffer("gather_index", layout.clamp_min(0), persistent=True)
        self.register_buffer("gather_mask", gather_mask.to(torch.float32), persistent=True)
        self.register_buffer(
            "position_encoding",
            _sinusoidal_positions(self.sequence_length, self.d_model),
            persistent=True,
        )

        self.token_projection = nn.Linear(BODY_FEATURE_WIDTH, self.d_model)
        self.q_projection = nn.Linear(self.d_model, self.d_model)
        self.k_projection = nn.Linear(self.d_model, self.d_model)
        self.v_projection = nn.Linear(self.d_model, self.d_model)
        self.attention_output = nn.Linear(self.d_model, self.d_model)
        self.ff_input = nn.Linear(self.d_model, ff_multiplier * self.d_model)
        self.ff_output = nn.Linear(ff_multiplier * self.d_model, self.d_model)

        # Affine-free LayerNorm and fixed positional/CLS inputs keep every trainable
        # encoder parameter inside an nn.Linear block, which makes the K-FAC
        # approximation explicit rather than silently mixing K-FAC and SGD params.
        self.pre_attention_norm = nn.LayerNorm(
            self.d_model, eps=layer_norm_eps, elementwise_affine=False
        )
        self.pre_ff_norm = nn.LayerNorm(
            self.d_model, eps=layer_norm_eps, elementwise_affine=False
        )
        self.output_norm = nn.LayerNorm(
            self.d_model, eps=layer_norm_eps, elementwise_affine=False
        )

        self.token_projection._kfac_num_locations = self.num_body_tokens
        for layer in (
            self.q_projection,
            self.k_projection,
            self.v_projection,
            self.attention_output,
            self.ff_input,
            self.ff_output,
        ):
            layer._kfac_num_locations = self.sequence_length

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        gathered = obs[..., self.gather_index]
        gathered = gathered * self.gather_mask.to(dtype=obs.dtype)
        return _token_linear(self.token_projection, gathered)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze_batch = obs.ndim == 1
        if squeeze_batch:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observations [batch,{self.obs_dim}], got {tuple(obs.shape)}"
            )

        body_tokens = self._tokenize(obs)
        cls_token = torch.zeros(
            obs.shape[0], 1, self.d_model, dtype=obs.dtype, device=obs.device
        )
        tokens = torch.cat((cls_token, body_tokens), dim=1)
        tokens = tokens + self.position_encoding.to(dtype=tokens.dtype)

        normalized = self.pre_attention_norm(tokens)
        query = _token_linear(self.q_projection, normalized)
        key = _token_linear(self.k_projection, normalized)
        value = _token_linear(self.v_projection, normalized)

        batch_size, seq_len, _ = query.shape
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        tokens = tokens + _token_linear(self.attention_output, context)

        ff_hidden = F.gelu(_token_linear(self.ff_input, self.pre_ff_norm(tokens)))
        tokens = tokens + _token_linear(self.ff_output, ff_hidden)
        cls_latent = self.output_norm(tokens)[:, 0]
        return cls_latent.squeeze(0) if squeeze_batch else cls_latent


def build_transformer(
    obs_space,
    env_name: str,
    nets_config=None,
    device="cuda",
):
    if nets_config is None:
        nets_config = SimpleNamespace(
            transformer_num_heads=4,
            transformer_ff_multiplier=2,
            transformer_num_layers=1,
            transformer_tokenization="mujoco_body",
            transformer_layer_norm_eps=1e-5,
        )

    tokenization = str(getattr(nets_config, "transformer_tokenization", "mujoco_body"))
    if tokenization != "mujoco_body":
        raise ValueError(f"Only transformer_tokenization=mujoco_body is supported, got {tokenization}")
    num_layers = int(getattr(nets_config, "transformer_num_layers", 1))
    if num_layers != 1:
        raise ValueError(f"This experiment requires a single Transformer layer, got {num_layers}")
    num_heads = int(getattr(nets_config, "transformer_num_heads", 4))
    ff_multiplier = int(getattr(nets_config, "transformer_ff_multiplier", 2))
    layer_norm_eps = float(getattr(nets_config, "transformer_layer_norm_eps", 1e-5))
    input_dim = int(obs_space.shape[0])

    def preprocess(obs_batch):
        obs_batch = np.asarray(obs_batch).astype(np.float32)
        return torch.from_numpy(obs_batch).to(device)

    def fn_neural_net(net_arch=None, p_dropout=0.0):
        if p_dropout != 0.0:
            raise ValueError("Transformer curvature experiments require dropout=0")
        if net_arch is None or len(net_arch) != 1:
            raise ValueError(
                "Single-layer Transformer expects exactly one hidden size in net_arch"
            )
        return SingleLayerBodyTransformer(
            obs_dim=input_dim,
            env_name=env_name,
            d_model=int(net_arch[0]),
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            layer_norm_eps=layer_norm_eps,
        )

    return fn_neural_net, preprocess
