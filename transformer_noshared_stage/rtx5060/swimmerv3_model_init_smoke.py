#!/usr/bin/env python3
"""Zero-learning M1/M2 Swimmer-v3 model-construction smoke test."""

import argparse
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--line", choices=("M1", "M2"), required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    from utils.mujoco_transformer import build_transformer
    from utils.utils import ActorCritic, build_mlp, count_vars

    config_name = (
        "rat_mlp_detjc_curv256_criticggn_actor_fvpclip05_batch262144.yaml"
        if args.line == "M1"
        else "rat_transformer_detjc_emp256_ggn256.yaml"
    )
    with (source / "configs" / config_name).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    algo_config = types.SimpleNamespace(**config["algo_config"])
    nets_config = types.SimpleNamespace(**config["nets_config"])

    env = gym.make("Swimmer-v3")
    obs, _ = env.reset(seed=123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if nets_config.type == "mlp":
        fn_neural_net, preprocess = build_mlp(env.observation_space, device=device)
    else:
        fn_neural_net, preprocess = build_transformer(
            env.observation_space,
            env_name="swimmer",
            nets_config=nets_config,
            device=device,
        )
    model = ActorCritic(
        fn_neural_net,
        env.observation_space.shape,
        nets_config=nets_config,
        dim_actions=env.action_space.shape[0],
        with_popart=algo_config.with_popart,
        sigma_type=algo_config.sigma_type,
        device=device,
    ).to(device)
    obs_tensor = preprocess(np.asarray([obs], dtype=np.float32))
    with torch.no_grad():
        values, policy_outputs = model(obs_tensor)

    print(f"line={args.line}")
    print(f"config={config_name}")
    print(f"network_type={nets_config.type}")
    print(f"device={device}")
    print(f"observation_shape={env.observation_space.shape}")
    print(f"action_shape={env.action_space.shape}")
    print(f"value_output_shape={tuple(values.shape)}")
    print(f"policy_output_shape={tuple(policy_outputs.shape)}")
    print(f"parameter_count={int(count_vars(model))}")
    print("optimizer_constructed=false")
    print("learning_steps=0")
    print(f"{args.line}_MODEL_INIT_SMOKE_OK")
    env.close()


if __name__ == "__main__":
    main()
