import os
import json
import yaml
import time
import math
import types
import torch
import argparse
import numpy as np
from tqdm import trange
from datetime import datetime
from collections import deque
import utils.logger as logger

from torch.nn import functional as F
from torch.func import vmap, grad, functional_call

# pytorch distributed training
import torch.multiprocessing as mp

from utils.runners import Runner
from utils.cg import conjugate_gradient
from utils.sketching import normalize_score_kernel, score_kernel, solve_score_kernel_system
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_mlp
from utils.utils import ActorCritic, count_vars, safemean, set_grads_from_flat, set_seed
from vec_env import VecNormalize

def learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps, nsteps, algo_config, log_config, log_dir=None):

    gamma = .999
    lam = .95

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    pi_minibatch_size = per_epoch_timesteps // algo_config.pi_minibatches
    v_minibatch_size = per_epoch_timesteps // algo_config.v_minibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    params_pi = [p for p in actor_critic.pi_net.parameters() if p.requires_grad]
    params_v = [p for p in actor_critic.v_net.parameters() if p.requires_grad]
    # Functional copies of the policy state are used for per-sample gradients.
    # They should be treated as read-only snapshots inside the update rules.
    dict_params = {k: v.detach() for k, v in actor_critic.pi_net.named_parameters() if v.requires_grad}
    dict_buffers = {k: v.detach() for k, v in actor_critic.pi_net.named_buffers() if v.requires_grad}
    dict_params_v = {k: v.detach() for k, v in actor_critic.v_net.named_parameters() if v.requires_grad}
    dict_buffers_v = {k: v.detach() for k, v in actor_critic.v_net.named_buffers() if v.requires_grad}

    if algo_config.optimizer == 'adam':
        pi_optimizer = Adam(params_pi, lr=algo_config.lr_pi, weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'sgd': 
        actor_sgd_momentum = float(getattr(algo_config, 'actor_sgd_momentum', 1e-6))
        pi_optimizer = SGD(params_pi, lr=algo_config.lr_pi, momentum=actor_sgd_momentum)
    elif algo_config.optimizer == 'kfac':
        from kfac.kfac import KFACOptimizer
        pi_optimizer = KFACOptimizer(actor_critic.pi_net, lr=algo_config.lr_pi,
                                     weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'ekfac':
        from kfac.ekfac import EKFACOptimizer
        pi_optimizer = EKFACOptimizer(actor_critic.pi_net, lr=algo_config.lr_pi,
                                     weight_decay=algo_config.weight_decay)
    else:
        raise NotImplementedError
    
    if hasattr(algo_config, 'lr_decay') and algo_config.lr_decay == 'cosine':
        pi_scheduler = CosineAnnealingLR(pi_optimizer, T_max=epochs*algo_config.pi_epochs*algo_config.pi_minibatches, eta_min=0.01)
    else:
        pi_scheduler = None

    # In this variant the critic also gets a sample-space Gauss-Newton step.
    # Use SGD semantics so lr_v directly scales the solved parameter direction.
    critic_optimizer_name = str(getattr(algo_config, 'critic_optimizer', 'sgd') or 'sgd').lower()
    if critic_optimizer_name == 'adam':
        v_optimizer = torch.optim.Adam(params_v, lr=algo_config.lr_v)
    elif critic_optimizer_name == 'sgd':
        critic_sgd_momentum = float(getattr(algo_config, 'critic_sgd_momentum', 1e-6))
        v_optimizer = SGD(params_v, lr=algo_config.lr_v, momentum=critic_sgd_momentum)
    else:
        raise ValueError(f"Unsupported critic_optimizer: {critic_optimizer_name}")

    # for trust region
    make_flat = lambda x:  torch.cat([grad.contiguous().view(-1) for grad in x if grad is not None])
    get_flat_grad = lambda params:  torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])

    actor_clip_mode = str(getattr(algo_config, 'actor_clip_mode', 'l2') or 'l2').lower()
    if actor_clip_mode not in {'l2', 'fvp_fisher'}:
        raise ValueError(f"Unsupported actor_clip_mode: {actor_clip_mode}")
    print(
        f"Actor clip mode: {actor_clip_mode}; max_grad_norm={algo_config.max_grad_norm}; "
        "FVP includes cg_damping",
        flush=True,
    )


    full_update_loss_state = {"records": 0, "stop": False}

    def selected_action_logp_from_outputs(outputs, actions):
        if actor_critic.is_discrete:
            logp_full = F.log_softmax(outputs, dim=-1)
            return torch.gather(logp_full, dim=-1, index=actions.unsqueeze(-1)).squeeze(1)
        mu, logstd = outputs.chunk(2, dim=-1)
        dist = torch.distributions.Normal(mu, torch.exp(logstd))
        return dist.log_prob(actions).sum(dim=-1)

    def full_batch_diagnostic_advantage(advantage, ratio=None):
        diagnostic_adv = advantage.to(dtype=torch.float32)
        if algo == 'ppo':
            return (diagnostic_adv - diagnostic_adv.mean()) / (diagnostic_adv.std() + 1e-8)
        diagnostic_adv = diagnostic_adv - diagnostic_adv.mean()
        if algo_config.norm_obj == 'adv':
            rms_sqrt = torch.sqrt(diagnostic_adv.pow(2).mean()).detach()
        elif algo_config.norm_obj == 'obj':
            ratio_for_norm = torch.ones_like(diagnostic_adv) if ratio is None else ratio.to(diagnostic_adv.dtype)
            rms_sqrt = torch.sqrt((ratio_for_norm * diagnostic_adv).pow(2).mean()).detach()
        elif algo_config.norm_obj == 'ratio':
            ratio_for_norm = torch.ones_like(diagnostic_adv) if ratio is None else ratio.to(diagnostic_adv.dtype)
            rms_sqrt = ratio_for_norm.mean().detach() * torch.sqrt(diagnostic_adv.pow(2).mean()).detach()
        else:
            raise NotImplementedError
        return diagnostic_adv / (rms_sqrt + 1e-8)

    def full_batch_policy_losses_for_diagnostic(obs_batch, act_batch, adv_batch, outputs_old_batch):
        outputs = actor_critic.forward_pi(obs_batch)
        logp = selected_action_logp_from_outputs(outputs, act_batch)
        logp_old = selected_action_logp_from_outputs(outputs_old_batch, act_batch).detach()
        ratio = torch.exp(logp - logp_old)
        if algo == 'ppo':
            diag_adv = full_batch_diagnostic_advantage(adv_batch, ratio=ratio)
            clip_adv = torch.clamp(ratio, 1 - algo_config.cliprange, 1 + algo_config.cliprange) * diag_adv
            losses = torch.max(-ratio * diag_adv, -clip_adv)
        else:
            if bool(getattr(algo_config, 'clamp_ratio', False)):
                ratio = torch.clamp(ratio, algo_config.min_ratio, algo_config.max_ratio)
            diag_adv = full_batch_diagnostic_advantage(adv_batch, ratio=ratio)
            losses = -ratio * diag_adv
        return losses, logp, ratio, diag_adv

    def make_full_update_loss_cache(obs_batch, act_batch, adv_batch, outputs_old_batch):
        if not getattr(algo_config, 'full_update_loss_log_path', None):
            return None
        if full_update_loss_state['stop']:
            return None
        start_epoch = getattr(algo_config, 'full_update_loss_start_epoch', None)
        if start_epoch is not None and epoch < int(start_epoch):
            return None
        if algo not in {'ppo', 'rat', 'fvp', 'diag', 'kfac', 'ekfac'}:
            return None
        with torch.no_grad():
            losses_before, logp_before, ratio_before, diag_adv = full_batch_policy_losses_for_diagnostic(obs_batch, act_batch, adv_batch, outputs_old_batch)
        return {'losses_before': losses_before.detach(), 'logp_before': logp_before.detach(), 'ratio_before': ratio_before.detach(), 'advantage': diag_adv.detach()}

    def log_full_update_loss_reduction(cache, obs_batch, act_batch, adv_batch, outputs_old_batch, epoch=None):
        log_path = getattr(algo_config, 'full_update_loss_log_path', None)
        if not log_path or cache is None or full_update_loss_state['stop']:
            return
        max_records = int(getattr(algo_config, 'full_update_loss_log_max_records', 0) or 0)
        remaining = None if max_records <= 0 else max(0, max_records - full_update_loss_state['records'])
        if remaining == 0:
            full_update_loss_state['stop'] = bool(getattr(algo_config, 'full_update_loss_stop_after_max', False))
            return
        with torch.no_grad():
            losses_after, logp_after, ratio_after, _ = full_batch_policy_losses_for_diagnostic(obs_batch, act_batch, adv_batch, outputs_old_batch)
            losses_before = cache['losses_before'].to(losses_after.device)
            logp_before = cache['logp_before'].to(losses_after.device)
            ratio_before = cache['ratio_before'].to(losses_after.device)
            diag_adv = cache['advantage'].to(losses_after.device)
            delta = losses_before - losses_after
            adv_sq = diag_adv.square()
            delta_over_adv2 = delta / (adv_sq + 1e-12)
            median_delta_over_adv2 = delta_over_adv2.median()
            median_scale = median_delta_over_adv2.abs().clamp_min(1e-12)
            signed_median = torch.where(median_delta_over_adv2 >= 0, median_scale, -median_scale)
            delta_over_adv2_over_median = delta_over_adv2 / signed_median
            log_abs_ratio = torch.log10((delta_over_adv2.abs() / median_scale).clamp_min(1e-12))
            count = delta.numel() if remaining is None else min(delta.numel(), remaining)
            records = []
            for idx in range(count):
                records.append({
                    'epoch': int(epoch) if epoch is not None else None,
                    'sample_index': int(idx),
                    'diagnostic_type': f'{algo}_full_actor_update_policy_loss_reduction',
                    'loss_type': f'{algo}_full_update_policy_surrogate',
                    'sample_loss_before': float(losses_before[idx].item()),
                    'sample_loss_after': float(losses_after[idx].item()),
                    'sample_loss_delta': float(delta[idx].item()),
                    'logp_before': float(logp_before[idx].item()),
                    'logp_after': float(logp_after[idx].item()),
                    'logp_delta': float((logp_after[idx] - logp_before[idx]).item()),
                    'ratio_before': float(ratio_before[idx].item()),
                    'ratio_after': float(ratio_after[idx].item()),
                    'advantage': float(diag_adv[idx].item()),
                    'advantage_sq': float(adv_sq[idx].item()),
                    'sample_loss_delta_over_adv2': float(delta_over_adv2[idx].item()),
                    'median_sample_loss_delta_over_adv2': float(median_delta_over_adv2.item()),
                    'sample_loss_delta_over_adv2_over_median': float(delta_over_adv2_over_median[idx].item()),
                    'log_abs_sample_loss_delta_over_adv2_over_median': float(log_abs_ratio[idx].item()),
                })
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        full_update_loss_state['records'] += len(records)
        if max_records > 0 and full_update_loss_state['records'] >= max_records:
            full_update_loss_state['stop'] = bool(getattr(algo_config, 'full_update_loss_stop_after_max', False))

    # Start total timer
    tfirststart = time.perf_counter()

    def TrustRegion_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # Build a Fisher-vector product and solve for the natural-gradient step.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _outputs_ref = _outputs.detach()
            _logp_full_ref = F.log_softmax(_outputs_ref, dim=-1)
            full_llr = _logp_full_ref - _logp_full
            _ent_kl = (torch.exp(_logp_full_ref) * full_llr).sum(dim=-1).mean()
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)

            _entropy = _dist.entropy().sum(dim=-1).mean()
            _outputs_ref = _outputs.detach()

            _mu_ref, _logstd_ref = _outputs_ref.chunk(2, dim=-1)
            _ent_kl = (_logstd - _logstd_ref + 0.5 * ( torch.exp(_logstd_ref).pow(2) + (_mu_ref - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)
        _loss_pi = (- _ratio * _adv).mean() 

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError

        # normalize the loss to stabilize the training
        _loss_pi = _loss_pi / (_rms_sqrt + 1e-8)

        # The entropy-based KL surrogate gives the curvature term for CG.
        kl_grad = torch.autograd.grad(_ent_kl, params_pi, create_graph=True)
        kl_grad_flat = make_flat(kl_grad)
        def fisher_vector_product(x):
            dot_prod = torch.dot(kl_grad_flat, x)
            fvp = torch.autograd.grad(dot_prod, params_pi, retain_graph=True)
            fvp_flat = make_flat(fvp)
            return fvp_flat + algo_config.cg_damping * x

        # Write the solved step into gradients so the optimizer can apply it.
        pi_optimizer.zero_grad()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward(retain_graph=True)

        if algo_config.grad == 'pg':
            step_dir = get_flat_grad(params_pi).detach()
        elif algo_config.grad == 'npg':
            loss_grad_pi_flat = get_flat_grad(params_pi).detach()
            step_dir = conjugate_gradient(fisher_vector_product, loss_grad_pi_flat, nsteps=algo_config.cg_steps)
        else:
            raise NotImplementedError

        ## gradient clipping
        max_grad_norm = algo_config.max_grad_norm
        if algo_config.post_grad == 'fisher_clip':
            grad_norm = torch.dot(step_dir, fisher_vector_product(step_dir)) # Fisher norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir * torch.clamp(max_grad_norm / grad_norm, max=1.0)
        elif algo_config.post_grad == 'l2_clip':
            grad_norm = torch.dot(step_dir, step_dir) # L2 norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir * torch.clamp(max_grad_norm / grad_norm, max=1.0)
        elif algo_config.post_grad == 'norm':
            grad_norm = torch.dot(step_dir, fisher_vector_product(step_dir)) # Fisher norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir / grad_norm.sqrt()
        else:
            grad_norm = torch.tensor(max_grad_norm) # no clipping

        set_grads_from_flat(params_pi, step_dir)

        pi_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, ent_kl=_ent_kl.item(),
                           kl_grad_norm=kl_grad_flat.norm().item(), grad_norm=grad_norm.item(),
                           ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info


    def PPO_ActorUpdate(_obs, _act, _adv, _outputs_old):
        pi_optimizer.zero_grad()
        _outputs = actor_critic.forward_pi(_obs)
        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _logp_old = torch.gather(_logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_logp - _logp_old)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = -_p_log_p.sum(-1).mean()
            _kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()
        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1)
            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)
            _ratio = torch.exp(_logp - _logp_old)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _kl = (_logstd - _logstd_old + 0.5 * (torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2)) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()
        _adv = (_adv - _adv.mean()) / (_adv.std() + 1e-8)
        _clip_adv = torch.clamp(_ratio, 1 - algo_config.cliprange, 1 + algo_config.cliprange) * _adv
        _losses_pi = torch.max(-_ratio * _adv, -_clip_adv)
        _loss_pi = _losses_pi.mean()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm)
        pi_optimizer.step()
        if pi_scheduler is not None:
            pi_scheduler.step()
        with torch.no_grad():
            clipped = _ratio.gt(1 + algo_config.cliprange) | _ratio.lt(1 - algo_config.cliprange)
            pi_info = dict(kl=_kl.item(), ent=_entropy.item(), cf=torch.as_tensor(clipped, dtype=torch.float32).mean().item(), curr_lr=pi_optimizer.param_groups[0]['lr'], grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())
        return _loss, _loss_pi, pi_info

    def RAT_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # RAT solves a sample-space system before mapping the step back to parameters.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_logp_full = F.log_softmax(batch_outs, dim=-1)
                batch_logp = torch.gather(batch_logp_full, dim=-1, index=batch_act.unsqueeze(-1)).squeeze(1)
                return batch_logp.squeeze(0)

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_mu, batch_logstd = batch_outs.chunk(2, dim=-1)

                var = torch.exp(batch_logstd)**2
                batch_logp = (
                    -((batch_act - batch_mu) ** 2) / (2 * var)
                    - batch_logstd
                    - math.log(math.sqrt(2 * math.pi))
                )

                return batch_logp.sum(dim=-1).squeeze(0)

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError
        _adv = _adv / (_rms_sqrt + 1e-8)

        pi_optimizer.zero_grad()
        # Per-sample policy gradients are required to form the batch-space linear system.
        # Grouped-rho uses the full minibatch, but reduces its 1024 sample
        # coefficients to 256 group multipliers (four samples per group).
        actor_group_size = int(getattr(algo_config, 'actor_group_size', 0) or 0)
        use_grouped_rho = actor_group_size > 0
        actor_curvature_subsample = getattr(algo_config, 'actor_curvature_subsample', 0) or 0
        if use_grouped_rho:
            actor_curvature_inds = None
            _actor_obs = _obs
            _actor_act = _act
            _actor_adv = _adv
            _actor_ratio = _ratio
            _actor_logp = _logp
        elif actor_curvature_subsample > 0 and _obs.shape[0] > actor_curvature_subsample:
            actor_curvature_inds = torch.randperm(_obs.shape[0], device=_obs.device)[:actor_curvature_subsample]
            _actor_obs = _obs[actor_curvature_inds]
            _actor_act = _act[actor_curvature_inds]
            _actor_adv = _adv[actor_curvature_inds]
            _actor_ratio = _ratio[actor_curvature_inds]
            _actor_logp = _logp[actor_curvature_inds]
        else:
            actor_curvature_inds = None
            _actor_obs = _obs
            _actor_act = _act
            _actor_adv = _adv
            _actor_ratio = _ratio
            _actor_logp = _logp

        actor_free_rho_mode = str(getattr(algo_config, 'actor_subsample_free_rho', 'none') or 'none').lower()
        free_rho_value = 0.0
        grouped_rho_std = 0.0
        grouped_rho_min = 0.0
        grouped_rho_max = 0.0
        use_strict_full_gradient_rho = (
            actor_free_rho_mode == 'strict'
            and actor_curvature_inds is not None
        )
        use_free_rho = (
            actor_free_rho_mode in {'anchor', 'full', 'one', 'zero'}
            and actor_curvature_inds is not None
        )

        ft_compute_sample_grad = vmap(grad(compute_logp), in_dims=(None, None, 0, 0))
        if use_grouped_rho:
            ft_per_sample_grads_full = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act)
            ft_per_sample_grads = ft_per_sample_grads_full
        elif use_free_rho:
            ft_per_sample_grads_full = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act)
            ft_per_sample_grads = {
                k: v.index_select(0, actor_curvature_inds)
                for k, v in ft_per_sample_grads_full.items()
            }
        else:
            ft_per_sample_grads_full = None
            ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _actor_obs, _actor_act) # num_samples x param_shape

        with torch.no_grad():
            num_sa = _actor_obs.shape[0]
            H = torch.cat([v.contiguous().view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)  # num_samples x num_params
            if use_grouped_rho:
                full_num_sa = _obs.shape[0]
                H_full = H
            elif use_free_rho:
                full_num_sa = _obs.shape[0]
                H_full = torch.cat([
                    v.contiguous().view(full_num_sa, -1)
                    for v in ft_per_sample_grads_full.values()
                ], dim=-1)
            else:
                full_num_sa = None
                H_full = None

            previous_projection = None
            g_k = None
            gk_list = [ v['momentum_buffer'].contiguous().flatten() for v in pi_optimizer.state.values() if v['momentum_buffer'] is not None ]
            if algo_config.is_karzmarz and len(gk_list) > 0:
                g_k = torch.cat(gk_list, dim=0)
                previous_projection = torch.mv(H, g_k)

            use_actor_identity_kernel = bool(getattr(algo_config, 'actor_identity_kernel', False))
            use_full_gradient_rhs = (
                bool(getattr(algo_config, 'actor_subsample_full_batch_gradient', False))
                and actor_curvature_inds is not None
            )
            effective_full_gradient_rhs = (
                use_full_gradient_rhs
                or use_strict_full_gradient_rho
            )
            if use_actor_identity_kernel:
                # Identity-kernel actor baseline: no Fisher/kernel inverse and no damping.
                # This is the ratio-weighted minibatch policy-gradient direction.
                _png_adv = _actor_adv.detach()
            elif effective_full_gradient_rhs:
                fisher_kernel = str(getattr(algo_config, 'fisher_kernel', 'exact') or 'exact').lower()
                if fisher_kernel != 'exact':
                    raise NotImplementedError(
                        "actor_subsample_full_batch_gradient currently supports fisher_kernel=exact"
                    )
                _png_adv = None
            elif use_free_rho:
                _png_adv = None
            else:
                _png_adv = solve_score_kernel_system(
                    H,
                    _actor_adv,
                    algo_config.cg_damping,
                    ratio=_actor_ratio,
                    kernel=algo_config.fisher_kernel,
                    sketch_dim=algo_config.sketch_dim,
                    sketch_seed=algo_config.sketch_seed,
                    diagonal_mode=algo_config.sketch_diagonal_mode,
                    normalization=algo_config.fisher_kernel_normalization,
                    normalization_eps=algo_config.fisher_kernel_normalization_eps,
                    previous_projection=previous_projection,
                )

        if actor_clip_mode == 'fvp_fisher' and (
            use_grouped_rho
            or use_free_rho
            or effective_full_gradient_rhs
        ):
            raise ValueError("actor_clip_mode=fvp_fisher does not support grouped-rho, free-rho, or full-gradient-RHS modes")

        # Update actor. fvp_fisher reproduces the original
        # TrustRegion_ActorUpdate clipping semantics exactly:
        #   q = g^T (F + damping I) g
        #   g <- g * min(1, max_grad_norm / q)
        # The original implementation calls q a Fisher norm without sqrt(q).
        actor_l2_norm_pre = torch.tensor(float('nan'), device=_obs.device)
        actor_fvp_quadratic_pre = torch.tensor(float('nan'), device=_obs.device)
        actor_fvp_quadratic_post = torch.tensor(float('nan'), device=_obs.device)
        actor_clip_coef = torch.tensor(1.0, device=_obs.device)
        if use_grouped_rho:
            _loss_pi = (-_ratio.detach() * _logp * _adv.detach()).mean()
            pi_optimizer.zero_grad()
            _loss_pi.backward(retain_graph=float(getattr(algo_config, "ent_coef", 0.0)) != 0.0)
            with torch.no_grad():
                damping = float(algo_config.cg_damping)
                if damping <= 0.0:
                    raise ValueError("actor_group_size requires positive cg_damping")
                if str(getattr(algo_config, 'fisher_kernel', 'exact')).lower() != 'exact':
                    raise NotImplementedError("actor_group_size currently requires fisher_kernel=exact")
                if str(getattr(algo_config, 'fisher_kernel_normalization', 'none')).lower() != 'none':
                    raise ValueError("actor_group_size currently requires fisher_kernel_normalization=none")
                full_m = int(H.shape[0])
                if full_m % actor_group_size != 0:
                    raise ValueError(f"minibatch size {full_m} is not divisible by actor_group_size={actor_group_size}")
                num_groups = full_m // actor_group_size
                if actor_curvature_subsample != num_groups:
                    raise ValueError(
                        f"actor_group_size={actor_group_size} creates {num_groups} groups; "
                        f"actor_curvature_subsample must equal that group count"
                    )

                group_perm = torch.randperm(full_m, device=H.device)
                grouped_idx = group_perm.view(num_groups, actor_group_size)
                grouped_H = H.index_select(0, group_perm).view(num_groups, actor_group_size, -1)
                target_full = _adv.detach().to(device=H.device, dtype=H.dtype)
                ratio_full = _ratio.detach().to(device=H.device, dtype=H.dtype)
                if g_k is None:
                    residual_full = target_full
                else:
                    residual_full = target_full - torch.mv(
                        H, g_k.to(device=H.device, dtype=H.dtype)
                    )
                grouped_b = residual_full.index_select(0, group_perm).view(num_groups, actor_group_size)
                grouped_ratio = ratio_full.index_select(0, group_perm).view(num_groups, actor_group_size)

                actor_test = torch.einsum('grp,gr->gp', grouped_H, grouped_b)
                actor_trial = torch.einsum('grp,gr->gp', grouped_H, grouped_ratio * grouped_b)
                metric_eps = float(getattr(algo_config, 'actor_subsample_free_rho_eps', 1e-12))
                group_metric = grouped_b.pow(2).sum(dim=1).clamp_min(metric_eps)
                reduced_system = actor_test @ actor_trial.t() / float(num_groups)
                reduced_system = reduced_system + damping * torch.diag(group_metric)
                grouped_rho = torch.linalg.solve(reduced_system, group_metric)
                update_dir = actor_trial.t().matmul(grouped_rho) / float(num_groups)
                set_grads_from_flat(params_pi, -update_dir)
                free_rho_value = float(grouped_rho.mean().item())
                grouped_rho_std = float(grouped_rho.std(unbiased=False).item())
                grouped_rho_min = float(grouped_rho.min().item())
                grouped_rho_max = float(grouped_rho.max().item())

            _loss = _loss_pi
            if float(getattr(algo_config, "ent_coef", 0.0)) != 0.0:
                entropy_loss = -algo_config.ent_coef * _entropy
                entropy_loss.backward()
                _loss = _loss + entropy_loss
        elif use_free_rho and actor_free_rho_mode != 'strict':
            _loss_pi = (-_ratio.detach() * _logp * _adv.detach()).mean()
            pi_optimizer.zero_grad()
            _loss_pi.backward(retain_graph=float(getattr(algo_config, "ent_coef", 0.0)) != 0.0)
            with torch.no_grad():
                damping = float(algo_config.cg_damping)
                if damping <= 0.0:
                    raise ValueError("actor_subsample_free_rho requires positive cg_damping")
                if H_full is None or actor_curvature_inds is None:
                    raise RuntimeError("actor_subsample_free_rho requires actor_curvature_subsample < minibatch size")

                full_m = int(H_full.shape[0])
                basis_idx = actor_curvature_inds
                basis_mask = torch.zeros(full_m, device=H_full.device, dtype=torch.bool)
                basis_mask[basis_idx] = True
                nonbasis_idx = torch.arange(full_m, device=H_full.device)[~basis_mask]

                # Coefficient-space target. This branch solves an alpha-system,
                # like solve_score_kernel_system, so b is the normalized advantage
                # target itself. The policy ratio belongs to the kernel columns and
                # final mean-gradient map, not to b/m.
                target_full = _adv.detach().to(dtype=H_full.dtype)
                ratio_full = _ratio.detach().to(dtype=H_full.dtype)

                prev_response_full = None
                if g_k is not None:
                    prev_response_full = torch.mv(
                        H_full,
                        g_k.to(device=H_full.device, dtype=H_full.dtype),
                    )

                target_s = target_full.index_select(0, basis_idx)
                ratio_s = ratio_full.index_select(0, basis_idx)
                if prev_response_full is None:
                    prev_s = torch.zeros_like(target_s)
                else:
                    prev_s = prev_response_full.index_select(0, basis_idx)

                if nonbasis_idx.numel() > 0:
                    H_n = H_full.index_select(0, nonbasis_idx)
                    target_n = target_full.index_select(0, nonbasis_idx)
                    ratio_n = ratio_full.index_select(0, nonbasis_idx)
                    if prev_response_full is None:
                        prev_n = torch.zeros_like(target_n)
                    else:
                        prev_n = prev_response_full.index_select(0, nonbasis_idx)
                    b_n = target_n - prev_n
                    g_n_b_n = torch.mv(H_n.t(), ratio_n * b_n)
                else:
                    g_n_b_n = torch.zeros(H_full.shape[1], device=H_full.device, dtype=H_full.dtype)

                # Match solve_score_kernel_system: build the score kernel first,
                # apply optional row/correlation normalization before multiplying
                # the policy-ratio columns, then solve with cg_damping.
                base_k_ss = torch.mm(H, H.t()) / float(num_sa)
                k_ss, row_scale = normalize_score_kernel(
                    base_k_ss,
                    normalization=getattr(algo_config, 'fisher_kernel_normalization', 'none'),
                    eps=float(getattr(algo_config, 'fisher_kernel_normalization_eps', 1e-12)),
                )
                if row_scale is not None:
                    row_scale = row_scale.to(dtype=k_ss.dtype)
                k_ss = k_ss * ratio_s.to(dtype=k_ss.dtype).unsqueeze(0)

                eye = torch.eye(num_sa, device=H.device, dtype=k_ss.dtype)
                M = k_ss + damping * eye

                k_sn_b_n_raw = torch.mv(H, g_n_b_n.to(dtype=H.dtype)) / float(num_sa)
                if row_scale is None:
                    b_s_rhs = target_s - prev_s
                    k_sn_b_n = k_sn_b_n_raw.to(dtype=k_ss.dtype)
                else:
                    # Existing normalized solver scales fixed previous responses,
                    # not the advantage target itself.
                    b_s_rhs = target_s - row_scale.to(dtype=target_s.dtype) * prev_s
                    k_sn_b_n = row_scale * k_sn_b_n_raw.to(dtype=k_ss.dtype)

                u = torch.linalg.solve(M, b_s_rhs.to(dtype=k_ss.dtype))
                v = torch.linalg.solve(M, k_sn_b_n.to(dtype=k_ss.dtype))

                if row_scale is None:
                    alpha_u = u
                    alpha_v = v
                else:
                    alpha_u = row_scale * u
                    alpha_v = row_scale * v

                d0 = torch.mv(H.t(), ratio_s.to(dtype=H.dtype) * alpha_u.to(dtype=H.dtype)) / float(full_m)
                d1 = (
                    g_n_b_n
                    - torch.mv(H.t(), ratio_s.to(dtype=H.dtype) * alpha_v.to(dtype=H.dtype))
                ) / float(full_m)

                response_target_full = target_full
                if prev_response_full is not None:
                    response_target_full = response_target_full - prev_response_full

                denom_eps = float(getattr(algo_config, 'actor_subsample_free_rho_eps', 1e-12))
                if actor_free_rho_mode == 'anchor':
                    # Energy/Galerkin restricted solve with one learnable
                    # non-anchor base coefficient: alpha_N = rho * b_N.
                    # Given u=M^-1 b_S and v=M^-1 K_SN b_N, Schur solves
                    # the coupled anchor coefficients and scalar rho.
                    if nonbasis_idx.numel() > 0:
                        c_base = torch.dot(
                            g_n_b_n.to(dtype=H.dtype),
                            g_n_b_n.to(dtype=H.dtype),
                        ) / float(num_sa)
                        target_n_energy = torch.dot(
                            target_n.to(dtype=k_ss.dtype),
                            target_n.to(dtype=k_ss.dtype),
                        )
                        c_mu = c_base.to(dtype=k_ss.dtype) + damping * target_n_energy
                        schur_num = target_n_energy - torch.dot(k_sn_b_n.to(dtype=k_ss.dtype), u)
                        schur_den = c_mu - torch.dot(k_sn_b_n.to(dtype=k_ss.dtype), v)
                        schur_den_safe = torch.where(
                            schur_den >= 0,
                            schur_den.abs().clamp_min(denom_eps),
                            -schur_den.abs().clamp_min(denom_eps),
                        )
                        rho = schur_num / schur_den_safe
                    else:
                        rho = torch.zeros((), device=H.device, dtype=k_ss.dtype)
                elif actor_free_rho_mode == 'full':
                    r0 = torch.mv(H_full, d0.to(dtype=H_full.dtype))
                    r1 = torch.mv(H_full, d1.to(dtype=H_full.dtype))
                    rho = torch.dot(r1, response_target_full - r0) / torch.dot(r1, r1).clamp_min(denom_eps)
                elif actor_free_rho_mode == 'strict':
                    rho = torch.as_tensor(1.0 / damping, device=H.device, dtype=k_ss.dtype)
                elif actor_free_rho_mode == 'one':
                    rho = torch.as_tensor(1.0, device=H.device, dtype=k_ss.dtype)
                elif actor_free_rho_mode == 'zero':
                    rho = torch.zeros((), device=H.device, dtype=k_ss.dtype)
                else:
                    raise ValueError(f"Unsupported actor_subsample_free_rho mode: {actor_free_rho_mode}")

                update_dir = d0 + rho.to(dtype=d1.dtype) * d1
                free_rho_value = float(rho.detach().item())
                set_grads_from_flat(params_pi, -update_dir)

            _loss = _loss_pi
            if float(getattr(algo_config, "ent_coef", 0.0)) != 0.0:
                entropy_loss = -algo_config.ent_coef * _entropy
                entropy_loss.backward()
                _loss = _loss + entropy_loss
        elif effective_full_gradient_rhs:
            # Curvature is estimated from the sampled score rows H_S, while the
            # RHS is the full minibatch policy-loss gradient g_B.
            _loss_pi = (-_ratio.detach() * _logp * _adv.detach()).mean()
            pi_optimizer.zero_grad()
            _loss_pi.backward(retain_graph=float(getattr(algo_config, "ent_coef", 0.0)) != 0.0)
            full_policy_grad = get_flat_grad(params_pi).detach()
            with torch.no_grad():
                damping = float(algo_config.cg_damping)
                if damping <= 0.0:
                    raise ValueError("actor_subsample_full_batch_gradient requires positive cg_damping")
                if use_strict_full_gradient_rho:
                    free_rho_value = float(1.0 / damping)

                raw_fullgrad_fisher = bool(
                    getattr(algo_config, 'actor_subsample_full_batch_gradient_raw_fisher', False)
                )
                ratio_s = _actor_ratio.detach().to(device=H.device, dtype=H.dtype)
                residual_grad = full_policy_grad
                initial_grad = None
                if g_k is not None:
                    initial_grad = g_k.to(device=H.device, dtype=full_policy_grad.dtype)
                    h_initial = torch.mv(H, initial_grad.to(dtype=H.dtype))
                    if raw_fullgrad_fisher:
                        fisher_initial = torch.mv(H.t(), h_initial.to(dtype=H.dtype)) / float(num_sa)
                    else:
                        fisher_initial = torch.mv(
                            H.t(), ratio_s * h_initial.to(dtype=H.dtype)
                        ) / float(num_sa)
                    residual_grad = (
                        full_policy_grad
                        - fisher_initial.to(dtype=full_policy_grad.dtype)
                        - damping * initial_grad
                    )

                rhs = torch.mv(H, residual_grad.to(dtype=H.dtype))
                raw_K = torch.mm(H, H.t())
                eye = torch.eye(num_sa, device=H.device, dtype=raw_K.dtype)
                if raw_fullgrad_fisher:
                    # Old/raw full-gradient 256-Fisher path:
                    #   (H_S^T H_S / s + damping I)^-1 g_B.
                    beta = torch.linalg.solve(
                        raw_K + float(num_sa) * damping * eye,
                        rhs.to(dtype=raw_K.dtype),
                    )
                    correction = torch.mv(H.t(), beta.to(dtype=H.dtype))
                else:
                    # Public-code-aligned ratio-weighted path:
                    #   (H_S^T D_ratio H_S / s + damping I)^-1 g_B.
                    weighted_K = raw_K * ratio_s.to(dtype=raw_K.dtype).unsqueeze(0)
                    beta = torch.linalg.solve(
                        weighted_K + float(num_sa) * damping * eye,
                        rhs.to(dtype=raw_K.dtype),
                    )
                    correction = torch.mv(
                        H.t(), ratio_s * beta.to(dtype=H.dtype)
                    )
                precond_grad = (
                    residual_grad - correction.to(dtype=full_policy_grad.dtype)
                ) / damping
                if initial_grad is not None:
                    precond_grad = precond_grad + initial_grad
                set_grads_from_flat(params_pi, precond_grad)
            _loss = _loss_pi
            if float(getattr(algo_config, "ent_coef", 0.0)) != 0.0:
                entropy_loss = -algo_config.ent_coef * _entropy
                entropy_loss.backward()
                _loss = _loss + entropy_loss
        else:
            _loss_pi = (- _actor_ratio.detach() * _actor_logp * _png_adv).mean()
            pi_optimizer.zero_grad()
            _loss = _loss_pi - algo_config.ent_coef * _entropy
            if actor_clip_mode == 'fvp_fisher':
                if str(getattr(algo_config, 'fisher_kernel', 'exact')).lower() != 'exact':
                    raise ValueError("actor_clip_mode=fvp_fisher requires fisher_kernel=exact")
                if str(getattr(algo_config, 'fisher_kernel_normalization', 'none')).lower() != 'none':
                    raise ValueError("actor_clip_mode=fvp_fisher requires fisher_kernel_normalization=none")

                kl_grad = torch.autograd.grad(_real_kl, params_pi, create_graph=True)
                kl_grad_flat = make_flat(kl_grad)

                def fisher_vector_product(x):
                    dot_prod = torch.dot(kl_grad_flat, x)
                    fvp = torch.autograd.grad(dot_prod, params_pi, retain_graph=True)
                    return make_flat(fvp) + algo_config.cg_damping * x

                _loss.backward(retain_graph=True)
                step_dir = get_flat_grad(params_pi).detach()
                actor_l2_norm_pre = step_dir.norm()
                actor_fvp_quadratic_pre = torch.dot(step_dir, fisher_vector_product(step_dir))
                assert actor_fvp_quadratic_pre.item() >= 0.0
                actor_clip_coef = torch.clamp(
                    torch.as_tensor(algo_config.max_grad_norm, device=step_dir.device, dtype=step_dir.dtype)
                    / actor_fvp_quadratic_pre,
                    max=1.0,
                )
                for p in params_pi:
                    if p.grad is not None:
                        p.grad.mul_(actor_clip_coef)
                actor_fvp_quadratic_post = actor_fvp_quadratic_pre * actor_clip_coef.square()
                grad_norm = actor_fvp_quadratic_pre
            else:
                _loss.backward()
        if actor_clip_mode == 'l2':
            grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm)
        pi_optimizer.step()
        if pi_scheduler is not None:
            pi_scheduler.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, 
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item(),
                           actor_curvature_samples=(actor_curvature_subsample if use_grouped_rho else num_sa),
                           actor_gradient_samples=int(_obs.shape[0]),
                           actor_full_gradient_rhs=int(effective_full_gradient_rhs),
                           actor_free_rho_mode=actor_free_rho_mode,
                           actor_free_rho_value=free_rho_value,
                           actor_free_rho=int(use_free_rho or use_strict_full_gradient_rho or use_grouped_rho),
                           actor_group_size=actor_group_size,
                           actor_grouped_rho=int(use_grouped_rho),
                           actor_grouped_rho_std=grouped_rho_std,
                           actor_grouped_rho_min=grouped_rho_min,
                           actor_grouped_rho_max=grouped_rho_max,
                           actor_clip_mode=actor_clip_mode,
                           actor_l2_norm_pre=actor_l2_norm_pre.item(),
                           actor_fvp_quadratic_pre=actor_fvp_quadratic_pre.item(),
                           actor_fvp_quadratic_post=actor_fvp_quadratic_post.item(),
                           actor_clip_coef=actor_clip_coef.item())

        return _loss, _loss_pi, pi_info

    def diag_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # This variant keeps only the diagonal of the Fisher approximation.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_logp_full = F.log_softmax(batch_outs, dim=-1)
                batch_logp = torch.gather(batch_logp_full, dim=-1, index=batch_act.unsqueeze(-1)).squeeze(1)
                return batch_logp.squeeze(0)

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_mu, batch_logstd = batch_outs.chunk(2, dim=-1)

                var = torch.exp(batch_logstd)**2
                batch_logp = (
                    -((batch_act - batch_mu) ** 2) / (2 * var)
                    - batch_logstd
                    - math.log(math.sqrt(2 * math.pi))
                )

                return batch_logp.sum(dim=-1).squeeze(0)

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError
        _adv = _adv / (_rms_sqrt + 1e-8)

        pi_optimizer.zero_grad()
        # Reuse the same per-sample gradients, then collapse them to a diagonal preconditioner.
        ft_compute_sample_grad = vmap(grad(compute_logp), in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape

        with torch.no_grad():
            num_sa = _obs.shape[0]
            H = torch.cat([v.contiguous().view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)  # num_samples x num_params
            diag_fisher = torch.linalg.norm(H, dim=0).pow(2) / num_sa  # num_params

        # udpate actor
        _loss_pi = (- _ratio.detach() * _logp * _adv).mean() 
        pi_optimizer.zero_grad()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward()

        loss_grad_pi_flat = get_flat_grad(params_pi).detach()
        step_dir = loss_grad_pi_flat / (diag_fisher + algo_config.cg_damping)
        set_grads_from_flat(params_pi, step_dir)

        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm) 
        pi_optimizer.step()
        if pi_scheduler is not None:
            pi_scheduler.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, 
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    def KFAC_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # KFAC alternates between collecting Fisher statistics and applying the policy step.
        pi_optimizer.zero_grad()
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)

            _entropy = _dist.entropy().sum(dim=-1).mean()
            _kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        # if pi_optimizer.steps % pi_optimizer.TCov == 0:
        if pi_optimizer.steps % pi_optimizer.TInv == 0:
            # Compute fisher, see Martens 2014
            actor_critic.pi_net.zero_grad()
            pg_fisher_loss = - _logp.mean()
            pi_optimizer.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            pi_optimizer.acc_stats = False

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)
        _loss_pi = (- _ratio * _adv).mean() 

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError

        # normalize the loss to stabilize the training
        _loss_pi = _loss_pi / (_rms_sqrt + 1e-8)
        _loss = _loss_pi - algo_config.ent_coef * _entropy

        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm)
        pi_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            approx_kl = _kl.item()
            ent = _entropy.item()
            pi_info = dict(kl=approx_kl, ent=ent, curr_lr=pi_optimizer.param_groups[0]['lr'], cf=clipfrac,
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    def CriticGN_Update(_obs, _ret):
        _vals = actor_critic.forward_v(_obs)
        _residual = (_ret - _vals).detach()
        mb_loss_v = F.mse_loss(_vals, _ret)

        def compute_value(params, buffers, batch_obs):
            batch_obs = batch_obs.unsqueeze(0)
            batch_vals = functional_call(actor_critic.v_net, (params, buffers), (batch_obs,))
            return batch_vals.reshape(-1)[0]

        critic_curvature_subsample = getattr(algo_config, 'critic_curvature_subsample', 0) or 0
        if critic_curvature_subsample > 0 and _obs.shape[0] > critic_curvature_subsample:
            critic_curvature_inds = torch.randperm(_obs.shape[0], device=_obs.device)[:critic_curvature_subsample]
            _critic_obs = _obs[critic_curvature_inds]
            _critic_residual = _residual[critic_curvature_inds]
        else:
            _critic_obs = _obs
            _critic_residual = _residual

        ft_compute_value_grad = vmap(grad(compute_value), in_dims=(None, None, 0))
        ft_per_sample_v_grads = ft_compute_value_grad(dict_params_v, dict_buffers_v, _critic_obs)

        with torch.no_grad():
            num_samples = _critic_obs.shape[0]
            J = torch.cat([v.contiguous().view(num_samples, -1) for v in ft_per_sample_v_grads.values()], dim=-1)

            # solve_score_kernel_system builds K = J J^T / n, matching the actor RAT path.
            alpha = solve_score_kernel_system(
                J,
                _critic_residual,
                algo_config.cg_damping,
                ratio=None,
                kernel=algo_config.fisher_kernel,
                sketch_dim=algo_config.sketch_dim,
                sketch_seed=algo_config.sketch_seed,
                diagonal_mode=algo_config.sketch_diagonal_mode,
                normalization=algo_config.fisher_kernel_normalization,
                normalization_eps=algo_config.fisher_kernel_normalization_eps,
                previous_projection=None,
            )
            step_dir = torch.mv(J.t(), alpha) / num_samples

        v_optimizer.zero_grad(set_to_none=False)
        for p in params_v:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        set_grads_from_flat(params_v, -step_dir)
        grad_norm = torch.nn.utils.clip_grad_norm_(params_v, 5.0)
        v_optimizer.step()

        with torch.no_grad():
            v_info = dict(v_grad_norm=grad_norm.item(), v_step_norm=step_dir.norm().item(),
                          critic_curvature_samples=num_samples)

        return mb_loss_v, v_info

    def CriticStandard_Update(_obs, _ret):
        _vals = actor_critic.forward_v(_obs)
        mb_loss_v = F.mse_loss(_vals, _ret)
        v_optimizer.zero_grad(set_to_none=False)
        mb_loss_v.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_v, 5.0)
        v_optimizer.step()
        with torch.no_grad():
            v_info = dict(v_grad_norm=grad_norm.item(), v_step_norm=0.0,
                          critic_curvature_samples=0)
        return mb_loss_v, v_info

    critic_update_mode = str(getattr(algo_config, 'critic_update', 'gn') or 'gn').lower()
    if critic_update_mode in {'gn', 'jointcritic', 'gauss_newton'}:
        update_critic = CriticGN_Update
    elif critic_update_mode in {'standard', 'sgd', 'mse'}:
        update_critic = CriticStandard_Update
    else:
        raise ValueError(f"Unsupported critic_update mode: {critic_update_mode}")

    # Select the actor update rule for the configured algorithm variant.
    if algo in {'ppo'}:
        update_actor = PPO_ActorUpdate
    elif algo in {'fvp'}: 
        update_actor = TrustRegion_ActorUpdate
    elif algo in {'rat'}:
        update_actor = RAT_ActorUpdate
    elif algo in {'kfac', 'ekfac'}:
        update_actor = KFAC_ActorUpdate
    elif algo in {'diag'}:
        update_actor = diag_ActorUpdate
    else: 
        raise NotImplementedError

    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)
    compute_time = []

    for epoch in tepochs:
        tstart = time.perf_counter()

        tepochs.set_description('Stepping environment...')

        actor_critic.eval() # set to eval mode
        obs, ret, act, adv, outputs_old, epinfos = runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        tepochs.set_description('Minibatch training...')

        # pop art
        if actor_critic.with_popart:
            actor_critic.last_v_layer.update(ret) # update the mean/var
            ret = actor_critic.last_v_layer.normalize(ret)
            adv = actor_critic.last_v_layer.normalize(adv)

        if actor_critic.obs_rms is not None:
            actor_critic.obs_rms.training = True
            obs = actor_critic.obs_rms(obs) # norm obs for training
            actor_critic.obs_rms.training = False
            # Re-evaluate the old policy on normalized observations so the ratios
            # are computed in the same input space used for training.
            with torch.no_grad():
                outputs_old = actor_critic.forward_pi(obs)

        actor_critic.train()  # set to train mode
        full_update_loss_cache = make_full_update_loss_cache(obs, act, adv, outputs_old)
        actor_tstart = time.perf_counter()
        for _ in range(algo_config.pi_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, pi_minibatch_size):
                end = start + pi_minibatch_size
                mbinds = inds[start:end]
                mb_obs, mb_act, mb_adv, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], outputs_old[mbinds]
                mb_loss, mb_loss_pi, pi_info = update_actor(mb_obs, mb_act, mb_adv, mb_outputs_old)
        actor_tnow = time.perf_counter()
        actor_time_elapsed = actor_tnow - actor_tstart
        compute_time.append(actor_time_elapsed)
        log_full_update_loss_reduction(full_update_loss_cache, obs, act, adv, outputs_old, epoch=epoch)
        if full_update_loss_state['stop']:
            print(f"full_update_loss_stop records={full_update_loss_state['records']} path={getattr(algo_config, 'full_update_loss_log_path', None)}", flush=True)
            break

        # kl adaptive lr adjustment
        if algo_config.use_kl_adaptive_lr:
            curr_kl = pi_info['kl']
            if curr_kl > 0.008 * 2:
                pi_optimizer.param_groups[0]['lr'] = max(pi_optimizer.param_groups[0]['lr'] / 1.5, 1e-4)
            elif curr_kl < 0.008 / 2:
                pi_optimizer.param_groups[0]['lr'] = min(pi_optimizer.param_groups[0]['lr'] * 1.5, 5e-2)

        for _ in range(algo_config.v_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, v_minibatch_size):
                end = start + v_minibatch_size
                mbinds = inds[start:end]
                _obs, _ret = obs[mbinds], ret[mbinds]
                mb_loss_v, v_info = update_critic(_obs, _ret)

        tepochs.set_postfix(loss_pi=mb_loss_pi.item(), loss_v=mb_loss_v.item(), entropy=pi_info['ent'], kl=pi_info['kl'], cf=pi_info['cf'], lr=pi_info['curr_lr'])

        # clean GPU cache
        torch.cuda.empty_cache()

        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(per_epoch_timesteps / (tnow - tstart))

        if logger.get_dir() is not None and (epoch+1) % log_config.log_interval == 0:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            logger.logkv("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps)
            logger.logkv("misc/nupdates", epoch)
            logger.logkv("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size)
            logger.logkv("fps", fps)
            logger.logkv("loss_pi", mb_loss_pi.item())
            logger.logkv("loss_v", mb_loss_v.item())
            logger.logkv("ret_max", ret.max().item())
            logger.logkv("ret_min", ret.min().item())
            logger.logkv("ret_avg", ret.mean().item())
            logger.logkv("ret_med", ret.median().item())
            logger.logkv("ret_var", ret.var().item())
            logger.logkv("action_max", act.max().item())
            logger.logkv("action_min", act.min().item())
            logger.logkv("adv_max", adv.max().item())
            logger.logkv("adv_min", adv.min().item())
            logger.logkv("adv_avg", adv.mean().item())
            logger.logkv("adv_med", adv.median().item())
            logger.logkv("adv_var", adv.var().item())
            logger.logkv("entropy", pi_info['ent'])
            logger.logkv("lr_pi", pi_info['curr_lr'])
            logger.logkv("kl", pi_info['kl'])
            if algo in {'fvp'}:
                logger.logkv("ent_kl", pi_info['ent_kl'])
                logger.logkv("kl_grad_norm", pi_info['kl_grad_norm'])
            logger.logkv("grad_norm", pi_info['grad_norm'])
            if pi_info.get('actor_clip_mode') == 'fvp_fisher':
                logger.logkv("actor_l2_norm_pre", pi_info['actor_l2_norm_pre'])
                logger.logkv("actor_fvp_quadratic_pre", pi_info['actor_fvp_quadratic_pre'])
                logger.logkv("actor_fvp_quadratic_post", pi_info['actor_fvp_quadratic_post'])
                logger.logkv("actor_clip_coef", pi_info['actor_clip_coef'])
            logger.logkv("lr_v", v_optimizer.param_groups[0]['lr'])
            logger.logkv("v_grad_norm", v_info['v_grad_norm'])
            logger.logkv("v_step_norm", v_info['v_step_norm'])
            logger.logkv("clipfrac", pi_info['cf'])
            logger.logkv("ratio_max", pi_info['ratio_max'])
            logger.logkv("ratio_min", pi_info['ratio_min'])
            if 'actor_free_rho' in pi_info:
                logger.logkv("actor_free_rho", pi_info['actor_free_rho'])
                logger.logkv("actor_free_rho_value", pi_info['actor_free_rho_value'])
            if pi_info.get('actor_grouped_rho', 0):
                logger.logkv("actor_group_size", pi_info['actor_group_size'])
                logger.logkv("actor_grouped_rho_std", pi_info['actor_grouped_rho_std'])
                logger.logkv("actor_grouped_rho_min", pi_info['actor_grouped_rho_min'])
                logger.logkv("actor_grouped_rho_max", pi_info['actor_grouped_rho_max'])
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)

            logger.dumpkvs()

        # Log changes from update
        # writer.add_scalar('train/rewards', rew.sum(), epoch)
        if writer is not None:
            writer.add_scalar('train/kl', pi_info['kl'], epoch)
            if algo in {'fvp'}:
                writer.add_scalar("ent_kl", pi_info['ent_kl'], epoch)
                writer.add_scalar("kl_grad_norm", pi_info['kl_grad_norm'], epoch)
            writer.add_scalar("grad_norm", pi_info['grad_norm'], epoch)
            if pi_info.get('actor_clip_mode') == 'fvp_fisher':
                writer.add_scalar("actor_l2_norm_pre", pi_info['actor_l2_norm_pre'], epoch)
                writer.add_scalar("actor_fvp_quadratic_pre", pi_info['actor_fvp_quadratic_pre'], epoch)
                writer.add_scalar("actor_fvp_quadratic_post", pi_info['actor_fvp_quadratic_post'], epoch)
                writer.add_scalar("actor_clip_coef", pi_info['actor_clip_coef'], epoch)
            writer.add_scalar('train/clipfrac', pi_info['cf'], epoch)
            writer.add_scalar('train/entropy', pi_info['ent'], epoch)
            writer.add_scalar('train/lr_pi', pi_info['curr_lr'], epoch)
            writer.add_scalar('train/ratio_max', pi_info['ratio_max'], epoch)
            writer.add_scalar('train/ratio_min', pi_info['ratio_min'], epoch)
            writer.add_scalar('train/loss_pi', mb_loss_pi, epoch)
            writer.add_scalar('train/loss_v', mb_loss_v, epoch)
            writer.add_scalar('train/lr_v', v_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/v_grad_norm', v_info['v_grad_norm'], epoch)
            writer.add_scalar('train/v_step_norm', v_info['v_step_norm'], epoch)
            writer.add_scalar("train/ret_max", ret.max().item(), epoch)
            writer.add_scalar("train/ret_min", ret.min().item(), epoch)
            writer.add_scalar("train/ret_avg", ret.mean().item(), epoch)
            writer.add_scalar("train/ret_med", ret.median().item(), epoch)
            writer.add_scalar("train/ret_var", ret.var().item(), epoch)
            writer.add_scalar("train/act_max", act.max().item(), epoch)
            writer.add_scalar("train/act_min", act.min().item(), epoch)
            writer.add_scalar("train/adv_max", adv.max().item(), epoch)
            writer.add_scalar("train/adv_min", adv.min().item(), epoch)
            writer.add_scalar("train/adv_avg", adv.mean().item(), epoch)
            writer.add_scalar("train/adv_med", adv.median().item(), epoch)
            writer.add_scalar("train/adv_var", adv.var().item(), epoch)
            writer.add_scalar('train/eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('train/eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('misc/time_elapsed', tnow - tfirststart, epoch)
            writer.add_scalar("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps, epoch)
            writer.add_scalar("misc/nupdates", epoch)
            writer.add_scalar("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size, epoch)

    if log_dir is not None:
        # save checkpoints
        torch.save({'model_state_dict': actor_critic.state_dict(), }, f'{log_dir}/model.ckpt')
        import json
        with open(f'{log_dir}/time.json', 'w') as f:
            json.dump({'compute_time_array': compute_time, 
                       'average': np.mean(compute_time), 
                       'time_per_update': np.mean(compute_time)/(per_epoch_timesteps / pi_minibatch_size * algo_config.pi_epochs), 
                       'stderr': np.std(compute_time)/np.sqrt(len(compute_time)), 
                       'updates': len(compute_time)}, f)

def train_fn(rank, world_size, algo, seed, algo_config, env_config, nets_config, log_config, device=-1):
    # Serialize data into file:
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Random seed
    if seed is None:
        seed = np.random.randint(1e6) + 10000 * rank # different seeds for each process
    set_seed(seed, torch_deterministic=True)

    env_name = env_config.env_name
    num_envs = env_config.num_envs

    if env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                    'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        timesteps_per_proc = env_config.timesteps_per_proc

    elif 'atari' not in env_name:
        env_name, distribution_mode, start_level, num_levels = env_name.split('-')
        start_level, num_levels = int(start_level), int(num_levels)

        if distribution_mode == 'easy':
            timesteps_per_proc = env_config.timesteps_per_proc_easy
        elif distribution_mode == 'hard':
            timesteps_per_proc = env_config.timesteps_per_proc_hard

    if rank==0:
        actor_sub = getattr(algo_config, 'actor_curvature_subsample', 0) or 0
        critic_sub = getattr(algo_config, 'critic_curvature_subsample', 0) or 0
        subcurve_tag = f".asub_{actor_sub}.csub_{critic_sub}" if actor_sub or critic_sub else ""
        clip_momentum_tag = (
            f".aclip_{getattr(algo_config, 'actor_clip_mode', 'l2')}"
            f"_{algo_config.max_grad_norm}"
            f".mompi_{float(getattr(algo_config, 'actor_sgd_momentum', 1e-6))}"
            f".momv_{float(getattr(algo_config, 'critic_sgd_momentum', 1e-6))}"
        )
        if env_name in {'cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                        'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant'}:
            log_dir = f"logs/{algo}.jointcritic.karzmarz_{algo_config.is_karzmarz}.{nets_config.type}.a{nets_config.a_hidden_size}x{nets_config.a_num_layers}x{nets_config.a_dropout}e{algo_config.pi_epochs}x{algo_config.pi_minibatches}.c{nets_config.c_hidden_size}x{nets_config.c_num_layers}x{nets_config.c_dropout}e{algo_config.v_epochs}x{algo_config.v_minibatches}.{algo_config.grad}_{algo_config.post_grad}_{algo_config.max_grad_norm}.{algo_config.sigma_type}.damping_{algo_config.cg_damping}.lr_pi_{algo_config.lr_pi}.lr_v_{algo_config.lr_v}{clip_momentum_tag}{subcurve_tag}/{env_name}.{time_now}_{seed}"
        else:
            log_dir = f"logs/{algo}.jointcritic.karzmarz_{algo_config.is_karzmarz}.{nets_config.type}{'_bn' if nets_config.with_bn else ''}_{algo_config.pi_epochs}epoch.damping_{algo_config.cg_damping}.lr_pi_{algo_config.lr_pi}.lr_v_{algo_config.lr_v}{subcurve_tag}/{env_config.env_name}.{time_now}_{seed}"

        format_strs = ['csv', 'stdout'] 
        logger.configure(dir=log_dir, format_strs=format_strs)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = None
        writer = None
    
    if rank==0:
        logger.info("creating environment")

    if env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'invertedpendulum', 'inverteddoublependulum',
                      'hopper', 'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
        tag_name = {'cartpole': 'CartPole-v1', 'acrobot': 'Acrobot-v1', 'mountaincar': 'MountainCar-v0', 
                    'lunarlander': 'LunarLander-v2', 'carracing': 'CarRacing-v2', 'invertedpendulum': 'InvertedPendulum-v4',
                    'inverteddoublependulum': 'InvertedDoublePendulum-v4',
                    'hopper': 'Hopper-v4', 'halfcheetah': 'HalfCheetah-v4', 'walker2d': 'Walker2d-v4', 
                    'humanoid': 'Humanoid-v4', 'humanoidstandup': 'HumanoidStandup-v4', 'reacher': 'Reacher-v4', 
                    'swimmer': 'Swimmer-v3', 'ant': 'Ant-v4'}
        
        vec_env_type = getattr(env_config, "vec_env_type", "dummy")
        vec_env_kwargs = {}
        if vec_env_type == "subproc":
            vec_env_cls = SubprocVecEnv
            vec_env_kwargs["start_method"] = getattr(env_config, "vec_env_start_method", "forkserver")
        elif vec_env_type == "grouped_subproc":
            from utils.grouped_subproc_vec_env import GroupedSubprocVecEnv
            vec_env_cls = GroupedSubprocVecEnv
            vec_env_kwargs["num_workers"] = getattr(env_config, "vec_env_num_workers", 11)
            vec_env_kwargs["start_method"] = getattr(env_config, "vec_env_start_method", "forkserver")
        elif vec_env_type == "dummy":
            vec_env_cls = None
        else:
            raise ValueError(f"Unsupported vec_env_type: {vec_env_type}")
        if rank == 0:
            logger.info(f"MuJoCo/Gym VecEnv backend: {vec_env_type}")

        venv = make_vec_env(
            tag_name[env_name],
            n_envs=num_envs,
            env_kwargs={'continuous': False} if env_name == 'carracing' else {},
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
        )

    else:
        raise NotImplementedError

    if device == -1:
        if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
            device_type = "cuda"
        else:
            device_type = "cpu"
        
        device = torch.device(device_type) # Select best available device
    else:
        assert device >= 0
        device = f"cuda:{device}"

    obs_space = venv.observation_space

    # Create actor-critic module
    if nets_config.type == 'mlp':
        kwargs = {'device': device}
        fn_neural_nets, preprocess = build_mlp(obs_space, **kwargs)
        obs_shape = obs_space.shape

    else: 
        raise NotImplementedError

    act_num, act_dim = None, None
    try:
        act_num = venv.action_space.n
    except AttributeError:
        act_dim = venv.action_space.shape[0]

    actor_critic = ActorCritic(fn_neural_nets, obs_shape, nets_config=nets_config, n_actions=act_num, 
                            dim_actions=act_dim, with_popart=algo_config.with_popart, 
                            sigma_type=algo_config.sigma_type, device=device).to(device)

    venv = VecNormalize(venv=venv, norm_ret=env_config.norm_ret, obs_preprocess=preprocess) # img transform and reward normalization

    if rank==0:
        logger.info(f'Running on device: {device}')
        logger.info(f"training...")

        # Count variables
        var_counts = count_vars(actor_critic)
        logger.log(f'\nNumber of parameters: {var_counts}\n')

        # yaml.dump(args, open( f"{log_dir}/args.yaml", 'w' ))
        config = {'algo_config': algo_config.__dict__, 
                'env_config': env_config.__dict__, 
                'nets_config': nets_config.__dict__, 
                'log_config': log_config.__dict__}

        yaml.dump(config, open( f"{log_dir}/config.yaml", 'w' ))

    learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps=timesteps_per_proc, nsteps=env_config.nsteps, 
          algo_config=algo_config, log_config=log_config, log_dir=log_dir)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--config', type=str, default='true_mlp.yaml')
    parser.add_argument('--device', type=int, default=-1) # -1: use any available device
    parser.add_argument('--env_name', type=str, default=None) # -1: use any available device
    parser.add_argument('--n_proc', type=int, default=1) # distributed training: number of processes
    parser.add_argument('--port_num', type=int, default=29500) # distributed training: number of processes
    parser.add_argument('--a_dropout', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--a_hidden_size', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--a_num_layers', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--c_dropout', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--c_hidden_size', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--c_num_layers', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--norm_obj', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--optimizer', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--sigma_type', type=str, default=None, choices=['vector', 'mu_shared', 'separate', 'linear']) 
    parser.add_argument('--cg_damping', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--pi_epochs', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--timesteps_per_proc', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--lr_pi', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--grad', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--post_grad', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--actor_clip_mode', type=str, default=None,
                        choices=['l2', 'fvp_fisher'])
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--actor_sgd_momentum', type=float, default=None)
    parser.add_argument('--critic_sgd_momentum', type=float, default=None)
    parser.add_argument('--is_karzmarz', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fisher_kernel', type=str, default=None,
                        choices=['exact', 'countsketch', 'countsketch_random', 'random_countsketch'])
    parser.add_argument('--sketch_dim', type=int, default=None)
    parser.add_argument('--sketch_seed', type=int, default=None)
    parser.add_argument('--sketch_diagonal_mode', type=str, default=None,
                        choices=['z_only', 'row_rescale'])
    parser.add_argument('--fisher_kernel_normalization', type=str, default=None,
                        choices=['none', 'correlation', 'diag', 'diagonal', 'row_norm', 'sample_norm'])
    parser.add_argument('--fisher_kernel_normalization_eps', type=float, default=None)
    parser.add_argument('--curvature_subsample', type=int, default=None,
                        help='Set both actor and critic curvature samples to this value.')
    parser.add_argument('--actor_curvature_subsample', type=int, default=None,
                        help='Use at most this many actor samples for the policy RAT kernel.')
    parser.add_argument('--critic_curvature_subsample', type=int, default=None,
                        help='Use at most this many critic samples for the value Gauss-Newton kernel.')
    parser.add_argument('--critic_update', type=str, default=None, choices=['gn', 'jointcritic', 'gauss_newton', 'standard', 'sgd', 'mse'],
                        help='Value update mode. Default gn preserves train_detach_jointcritic.py behavior; standard uses ordinary MSE SGD.')
    parser.add_argument('--critic_optimizer', type=str, default=None, choices=['sgd', 'adam'],
                        help='Optimizer for critic parameters. Default sgd preserves jointcritic behavior; adam is for ordinary MSE critic baselines.')
    parser.add_argument('--actor_subsample_full_batch_gradient', action=argparse.BooleanOptionalAction, default=None,
                        help='Estimate actor Fisher with actor_curvature_subsample but precondition the full minibatch gradient.')
    parser.add_argument('--actor_identity_kernel', action=argparse.BooleanOptionalAction, default=None,
                        help='Use identity actor kernel: ratio-weighted minibatch policy gradient, no Fisher inverse/damping.')
    parser.add_argument('--actor_subsample_full_batch_gradient_raw_fisher', action=argparse.BooleanOptionalAction, default=None,
                        help='For actor_subsample_full_batch_gradient, use old/raw H_S^T H_S Fisher without policy-ratio weighting inside the inverse.')
    parser.add_argument('--actor_subsample_free_rho', type=str, default=None, choices=['none', 'anchor', 'full', 'strict', 'one', 'zero'],
                        help='Use free-rho non-anchor base for actor subsampled curvature; actor-only experimental mode')
    parser.add_argument('--actor_subsample_free_rho_eps', type=float, default=None,
                        help='Numerical epsilon for free-rho scalar denominator')
    parser.add_argument('--actor_group_size', type=int, default=None,
                        help='Tie alpha_i=rho_g*b_i within fixed-size random sample groups; group count is actor_curvature_subsample')
    parser.add_argument('--full_update_loss_log_path', type=str, default=None)
    parser.add_argument('--full_update_loss_log_max_records', type=int, default=None)
    parser.add_argument('--full_update_loss_stop_after_max', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--full_update_loss_start_epoch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None) 

    args = parser.parse_args()

    with open(f'configs/{args.config}') as fin:
        config = yaml.safe_load(fin)

    algo = config['algo']
    algo_config = types.SimpleNamespace(**config['algo_config'])
    env_config = types.SimpleNamespace(**config['env_config'])
    nets_config = types.SimpleNamespace(**config['nets_config'])
    log_config = types.SimpleNamespace(**config['log_config'])

    if args.env_name is not None:
        env_config.env_name = args.env_name

    if args.a_hidden_size is not None:
        nets_config.a_hidden_size = args.a_hidden_size
    if args.a_num_layers is not None:
        nets_config.a_num_layers = args.a_num_layers
    if args.a_dropout is not None:
        nets_config.a_dropout = args.a_dropout
    if args.c_hidden_size is not None:
        nets_config.c_hidden_size = args.c_hidden_size
    if args.c_num_layers is not None:
        nets_config.c_num_layers = args.c_num_layers
    if args.c_dropout is not None:
        nets_config.c_dropout = args.c_dropout

    if args.optimizer is not None:
        algo_config.optimizer = args.optimizer

    if args.sigma_type is not None:
        algo_config.sigma_type = args.sigma_type

    if args.norm_obj is not None:
        algo_config.norm_obj = args.norm_obj

    if args.cg_damping is not None:
        algo_config.cg_damping = args.cg_damping

    if args.pi_epochs is not None:
        algo_config.pi_epochs = args.pi_epochs

    if args.lr_pi is not None:
        algo_config.lr_pi = args.lr_pi

    if args.grad is not None:
        algo_config.grad = args.grad

    if args.post_grad is not None:
        algo_config.post_grad = args.post_grad

    if args.actor_clip_mode is not None:
        algo_config.actor_clip_mode = args.actor_clip_mode
    if args.max_grad_norm is not None:
        algo_config.max_grad_norm = args.max_grad_norm
    if args.actor_sgd_momentum is not None:
        algo_config.actor_sgd_momentum = args.actor_sgd_momentum
    if args.critic_sgd_momentum is not None:
        algo_config.critic_sgd_momentum = args.critic_sgd_momentum
    if args.is_karzmarz is not None:
        algo_config.is_karzmarz = args.is_karzmarz

    if args.timesteps_per_proc is not None:
        env_config.timesteps_per_proc = args.timesteps_per_proc

    if not hasattr(algo_config, 'fisher_kernel'):
        algo_config.fisher_kernel = 'exact'
    if not hasattr(algo_config, 'sketch_dim'):
        algo_config.sketch_dim = 32768
    if not hasattr(algo_config, 'sketch_seed'):
        algo_config.sketch_seed = 0
    if not hasattr(algo_config, 'sketch_diagonal_mode'):
        algo_config.sketch_diagonal_mode = 'z_only'
    if not hasattr(algo_config, 'fisher_kernel_normalization'):
        algo_config.fisher_kernel_normalization = 'correlation'
    if not hasattr(algo_config, 'fisher_kernel_normalization_eps'):
        algo_config.fisher_kernel_normalization_eps = 1e-12
    if not hasattr(algo_config, 'actor_curvature_subsample'):
        algo_config.actor_curvature_subsample = 0
    if not hasattr(algo_config, 'critic_curvature_subsample'):
        algo_config.critic_curvature_subsample = 0
    if not hasattr(algo_config, 'critic_update'):
        algo_config.critic_update = 'gn'
    if not hasattr(algo_config, 'critic_optimizer'):
        algo_config.critic_optimizer = 'sgd'
    if not hasattr(algo_config, 'actor_subsample_full_batch_gradient'):
        algo_config.actor_subsample_full_batch_gradient = False
    if not hasattr(algo_config, 'actor_identity_kernel'):
        algo_config.actor_identity_kernel = False
    if not hasattr(algo_config, 'actor_subsample_full_batch_gradient_raw_fisher'):
        algo_config.actor_subsample_full_batch_gradient_raw_fisher = False
    if not hasattr(algo_config, 'actor_subsample_free_rho'):
        algo_config.actor_subsample_free_rho = 'none'
    if not hasattr(algo_config, 'actor_subsample_free_rho_eps'):
        algo_config.actor_subsample_free_rho_eps = 1e-12
    if not hasattr(algo_config, 'actor_group_size'):
        algo_config.actor_group_size = 0

    if args.fisher_kernel is not None:
        algo_config.fisher_kernel = args.fisher_kernel
    if args.sketch_dim is not None:
        algo_config.sketch_dim = args.sketch_dim
    if args.sketch_seed is not None:
        algo_config.sketch_seed = args.sketch_seed
    if args.sketch_diagonal_mode is not None:
        algo_config.sketch_diagonal_mode = args.sketch_diagonal_mode
    if args.fisher_kernel_normalization is not None:
        algo_config.fisher_kernel_normalization = args.fisher_kernel_normalization
    if args.fisher_kernel_normalization_eps is not None:
        algo_config.fisher_kernel_normalization_eps = args.fisher_kernel_normalization_eps
    if args.curvature_subsample is not None:
        algo_config.actor_curvature_subsample = args.curvature_subsample
        algo_config.critic_curvature_subsample = args.curvature_subsample
    if args.actor_curvature_subsample is not None:
        algo_config.actor_curvature_subsample = args.actor_curvature_subsample
    if args.critic_curvature_subsample is not None:
        algo_config.critic_curvature_subsample = args.critic_curvature_subsample
    if args.critic_update is not None:
        algo_config.critic_update = args.critic_update
    if args.critic_optimizer is not None:
        algo_config.critic_optimizer = args.critic_optimizer
    if args.actor_subsample_full_batch_gradient is not None:
        algo_config.actor_subsample_full_batch_gradient = args.actor_subsample_full_batch_gradient
    if args.actor_identity_kernel is not None:
        algo_config.actor_identity_kernel = args.actor_identity_kernel
    if args.actor_subsample_full_batch_gradient_raw_fisher is not None:
        algo_config.actor_subsample_full_batch_gradient_raw_fisher = args.actor_subsample_full_batch_gradient_raw_fisher
    if args.actor_subsample_free_rho is not None:
        algo_config.actor_subsample_free_rho = args.actor_subsample_free_rho
    if args.actor_subsample_free_rho_eps is not None:
        algo_config.actor_subsample_free_rho_eps = args.actor_subsample_free_rho_eps
    if args.actor_group_size is not None:
        algo_config.actor_group_size = args.actor_group_size

    if args.full_update_loss_log_path is not None:
        algo_config.full_update_loss_log_path = args.full_update_loss_log_path
    if args.full_update_loss_log_max_records is not None:
        algo_config.full_update_loss_log_max_records = args.full_update_loss_log_max_records
    if args.full_update_loss_stop_after_max is not None:
        algo_config.full_update_loss_stop_after_max = args.full_update_loss_stop_after_max
    if args.full_update_loss_start_epoch is not None:
        algo_config.full_update_loss_start_epoch = args.full_update_loss_start_epoch

    if args.n_proc > 1:
        # multiple nodes
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port_num)

        mp.spawn(train_fn, args=(args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device),
                        nprocs=args.n_proc, # INFO: for TPU, either 1 or the maximum number of TPU chips
                        join=True)

    else:
        train_fn(0, args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device)

if __name__ == '__main__':
    main()
