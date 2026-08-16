from types import SimpleNamespace

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from kfac.kfac import KFACOptimizer
from utils.mujoco_transformer import (
    SingleLayerBodyTransformer,
    build_mujoco_body_layout,
    build_transformer,
)
from utils.utils import ActorCritic


ENV_SPECS = {
    "Ant-v4": (27, 8),
    "HalfCheetah-v4": (17, 6),
    "Hopper-v4": (11, 3),
    "Humanoid-v4": (376, 17),
    "HumanoidStandup-v4": (376, 17),
    "Swimmer-v3": (8, 2),
    "Walker2d-v4": (17, 6),
}


def _assert_finite(tree):
    for name, tensor in tree.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"Non-finite tensor: {name}")


def check_layouts_and_forward():
    for env_name, (obs_dim, _) in ENV_SPECS.items():
        layout, names = build_mujoco_body_layout(env_name, obs_dim)
        used = torch.sort(layout[layout >= 0]).values
        assert torch.equal(used, torch.arange(obs_dim))
        assert layout.shape[0] == len(names)

        model = SingleLayerBodyTransformer(obs_dim, env_name, d_model=64, num_heads=4)
        output = model(torch.randn(3, obs_dim))
        assert output.shape == (3, 64)
        assert torch.isfinite(output).all()


def check_actor_critic_independence():
    for env_name, (obs_dim, action_dim) in ENV_SPECS.items():
        obs_space = SimpleNamespace(shape=(obs_dim,))
        nets_config = SimpleNamespace(
            norm_obs=True,
            a_dropout=0.0,
            a_hidden_size=64,
            a_num_layers=1,
            c_dropout=0.0,
            c_hidden_size=64,
            c_num_layers=1,
            transformer_num_layers=1,
            transformer_num_heads=4,
            transformer_ff_multiplier=2,
            transformer_tokenization="mujoco_body",
            transformer_layer_norm_eps=1e-5,
        )
        builder, _ = build_transformer(
            obs_space,
            env_name=env_name,
            nets_config=nets_config,
            device="cpu",
        )
        actor_critic = ActorCritic(
            builder,
            obs_space.shape,
            nets_config=nets_config,
            dim_actions=action_dim,
            with_popart=True,
            sigma_type="vector",
            device="cpu",
        )
        pi_parameter_ids = {id(parameter) for parameter in actor_critic.pi_net.parameters()}
        value_parameter_ids = {id(parameter) for parameter in actor_critic.v_net.parameters()}
        assert pi_parameter_ids.isdisjoint(value_parameter_ids)

        values, policy = actor_critic(torch.randn(4, obs_dim))
        assert values.shape == (4,)
        assert policy.shape == (4, 2 * action_dim)
        assert torch.isfinite(values).all()
        assert torch.isfinite(policy).all()


def check_per_sample_gradients():
    model = SingleLayerBodyTransformer(17, "HalfCheetah-v4", d_model=64, num_heads=4)
    params = dict(model.named_parameters())
    observations = torch.randn(5, 17)

    def scalar_output(current_params, observation):
        output = functional_call(model, current_params, (observation.unsqueeze(0),))
        return output.square().mean()

    per_sample_grads = vmap(grad(scalar_output), in_dims=(None, 0))(params, observations)
    _assert_finite(per_sample_grads)
    for name, tensor in per_sample_grads.items():
        assert tensor.shape[0] == observations.shape[0], name


def check_kfac_step():
    model = nn.Sequential(
        SingleLayerBodyTransformer(17, "HalfCheetah-v4", d_model=64, num_heads=4),
        nn.Linear(64, 6),
    )
    optimizer = KFACOptimizer(
        model,
        lr=1e-2,
        damping=0.1,
        kl_clip=1e-3,
        TCov=1,
        TInv=1,
        batch_averaged=True,
    )
    optimizer.zero_grad()
    optimizer.acc_stats = True
    loss = model(torch.randn(8, 17)).square().mean()
    loss.backward()
    optimizer.acc_stats = False
    optimizer.step()

    assert len(optimizer.modules) == 8
    for module in optimizer.modules:
        assert module in optimizer.m_aa
        assert module in optimizer.m_gg
        assert torch.isfinite(optimizer.m_aa[module]).all()
        assert torch.isfinite(optimizer.m_gg[module]).all()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def main():
    torch.manual_seed(0)
    check_layouts_and_forward()
    check_actor_critic_independence()
    check_per_sample_gradients()
    check_kfac_step()
    print("TRANSFORMER_NOSHARED_VERIFY_OK")


if __name__ == "__main__":
    main()
