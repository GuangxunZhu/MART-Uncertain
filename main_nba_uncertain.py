import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import UNCERTAINMART as MART
from loaders.dataloader_nba import NBADataset
import torch
import torch.nn.functional as F
def compute_loss(
    y_pred,
    target,
    edge_uncertainty,
    sample_ids,
    best_edge_bank,
    epoch,
    best_start_epoch=20,
    best_warmup_epochs=20,
    lambda_best=0.1,
    lambda_nll=0.01,
    min_var=1e-4,
    max_var=1e3
):
    if target.dim() == 5 and target.size(2) == 1:
        target = target.squeeze(2)

    B, N, S, K, T, _ = y_pred.shape

    # --------------------------------------------------
    # 1. Prediction loss: Mean-S(Best-of-K)
    # --------------------------------------------------
    target_expand = target.unsqueeze(2).unsqueeze(3)
    displacement_error = torch.norm(y_pred - target_expand, dim=-1)
    decoder_ade = displacement_error.mean(dim=4)

    best_decoder_ade_per_sample, best_decoder_idx = decoder_ade.min(dim=3)
    loss_pred = best_decoder_ade_per_sample.mean()

    # --------------------------------------------------
    # 2. Current best sample for each agent
    # --------------------------------------------------
    best_sample_ade, best_sample_idx = best_decoder_ade_per_sample.min(dim=2)

    sampled_edges = edge_uncertainty["sampled_edges"]       # [B,N,N,S,D]
    mu_edge = edge_uncertainty["mu_edge"]                   # [B,N,N,D]
    total_var_edge = edge_uncertainty["total_var_edge"]

    D = sampled_edges.size(-1)

    gather_idx = best_sample_idx[:, :, None, None, None].expand(B, N, N, 1, D)
    current_best_edges = torch.gather(sampled_edges, dim=3, index=gather_idx).squeeze(3)
    # [B,N,N,D]

    # --------------------------------------------------
    # 3. Update historical best bank
    # --------------------------------------------------
    historical_best_edges = torch.zeros_like(current_best_edges)

    for b in range(B):
        scene_id = int(sample_ids[b])

        for agent_idx in range(N):
            key = (scene_id, agent_idx)

            current_score = float(best_sample_ade[b, agent_idx].detach().cpu())
            current_edge = current_best_edges[b, agent_idx].detach()

            if key not in best_edge_bank or current_score < best_edge_bank[key]["score"]:
                best_edge_bank[key] = {
                    "score": current_score,
                    "edge": current_edge.cpu().clone()
                }

            historical_best_edges[b, agent_idx] = best_edge_bank[key]["edge"].to(
                device=mu_edge.device,
                dtype=mu_edge.dtype
            )

    # --------------------------------------------------
    # 4. Historical best MSE
    # --------------------------------------------------
    edge_mask_4d = (~torch.eye(N, dtype=torch.bool, device=mu_edge.device)).view(1, N, N, 1)
    edge_mask_4d = edge_mask_4d.expand_as(mu_edge)

    loss_best = F.mse_loss(
        mu_edge[edge_mask_4d],
        historical_best_edges.detach()[edge_mask_4d]
    )

    # --------------------------------------------------
    # 5. Warm-up
    # --------------------------------------------------
    if epoch < best_start_epoch:
        best_weight = 0.0
    elif best_warmup_epochs > 0 and epoch < best_start_epoch + best_warmup_epochs:
        warmup_progress = (epoch - best_start_epoch + 1) / best_warmup_epochs
        best_weight = lambda_best * warmup_progress
    else:
        best_weight = lambda_best

    # --------------------------------------------------
    # 6. Gaussian NLL
    # --------------------------------------------------
    agent_residual = best_sample_ade.detach().pow(2)

    pair_residual = 0.5 * (
        agent_residual.unsqueeze(2)
        + agent_residual.unsqueeze(1)
    )

    edge_var_pair = total_var_edge.mean(dim=-1).clamp(
        min=min_var,
        max=max_var
    )

    edge_mask = ~torch.eye(N, dtype=torch.bool, device=mu_edge.device).unsqueeze(0)
    edge_mask = edge_mask.expand(B, N, N)

    edge_nll = 0.5 * (
        pair_residual / edge_var_pair
        + torch.log(edge_var_pair)
    )

    loss_edge_nll = edge_nll[edge_mask].mean()
    loss_nll = lambda_nll * loss_edge_nll

    # --------------------------------------------------
    # 7. Monitoring
    # --------------------------------------------------
    edge_var = total_var_edge[:, ~torch.eye(N, dtype=torch.bool, device=mu_edge.device), :].mean()

    pred_mean = y_pred.mean(dim=3)
    pred_var_mean = pred_mean.var(dim=2, unbiased=False).mean()

    best_decoder_idx_expand = best_decoder_idx[:, :, :, None, None, None].expand(B, N, S, 1, T, 2)
    best_pred_per_sample = torch.gather(y_pred, dim=3, index=best_decoder_idx_expand).squeeze(3)
    pred_var_best = best_pred_per_sample.var(dim=2, unbiased=False).mean()

    # --------------------------------------------------
    # 8. Total
    # --------------------------------------------------
    total_loss = loss_pred + best_weight * loss_best + loss_nll

    loss_dict = {
        "loss_pred": loss_pred.detach(),
        "loss_best": loss_best.detach(),
        "loss_edge_nll": loss_edge_nll.detach(),
        "loss_nll": loss_nll.detach(),
        "best_weight": best_weight,
        "edge_var": edge_var.detach(),
        "pair_residual": pair_residual[edge_mask].detach().mean(),
        "pred_var_mean": pred_var_mean.detach(),
        "pred_var_best": pred_var_best.detach(),
        "best_sample_ade": best_sample_ade.detach().mean()
    }

    return total_loss, loss_dict
# def compute_loss(
#     y_pred,
#     target,
#     edge_uncertainty,
#     epoch,
#     best_start_epoch=20,
#     best_warmup_epochs=20,
#     lambda_best=0.1,
#     lambda_nll=0.01,
#     min_var=1e-4,
#     max_var=1e3
# ):
#     if target.dim() == 5 and target.size(2) == 1:
#         target = target.squeeze(2)

#     B, N, S, K, T, _ = y_pred.shape

#     # --------------------------------------------------
#     # 1. Prediction loss: Mean-S(Best-of-K)
#     # --------------------------------------------------
#     target_expand = target.unsqueeze(2).unsqueeze(3)
#     displacement_error = torch.norm(y_pred - target_expand, dim=-1)
#     decoder_ade = displacement_error.mean(dim=4)

#     best_decoder_ade_per_sample, best_decoder_idx = decoder_ade.min(dim=3)
#     loss_pred = best_decoder_ade_per_sample.mean()

#     # --------------------------------------------------
#     # 2. Find best edge sample for each agent
#     # --------------------------------------------------
#     best_sample_ade, best_sample_idx = best_decoder_ade_per_sample.min(dim=2)

#     sampled_edges = edge_uncertainty["sampled_edges"]
#     mu_edge = edge_uncertainty["mu_edge"]
#     total_var_edge = edge_uncertainty["total_var_edge"]

#     gather_idx = best_sample_idx[:, :, None, None, None].expand(
#         B, N, N, 1, sampled_edges.size(-1)
#     )

#     best_edges = torch.gather(
#         sampled_edges,
#         dim=3,
#         index=gather_idx
#     ).squeeze(3)

#     # --------------------------------------------------
#     # 3. Best-edge MSE
#     # --------------------------------------------------
#     loss_best = F.mse_loss(
#         mu_edge,
#         best_edges.detach()
#     )

#     # --------------------------------------------------
#     # 4. Best-edge weight warm-up
#     # --------------------------------------------------
#     if epoch < best_start_epoch:
#         best_weight = 0.0
#     elif best_warmup_epochs > 0 and epoch < best_start_epoch + best_warmup_epochs:
#         warmup_progress = (epoch - best_start_epoch + 1) / best_warmup_epochs
#         best_weight = lambda_best * warmup_progress
#     else:
#         best_weight = lambda_best

#     # --------------------------------------------------
#     # 5. Gaussian NLL for edge variance
#     # --------------------------------------------------

#     # Agent-level residual target
#     # [B,N]
#     agent_residual = best_sample_ade.detach().pow(2)

#     # Convert agent residual to pair residual
#     # [B,N,N]
#     pair_residual = 0.5 * (
#         agent_residual.unsqueeze(2)
#         + agent_residual.unsqueeze(1)
#     )

#     # Reduce D-dimensional edge variance to scalar variance per edge
#     # [B,N,N,D] -> [B,N,N]
#     edge_var_pair = total_var_edge.mean(dim=-1)
#     edge_var_pair = edge_var_pair.clamp(
#         min=min_var,
#         max=max_var
#     )

#     # Ignore self edges
#     edge_mask = ~torch.eye(
#         N,
#         dtype=torch.bool,
#         device=total_var_edge.device
#     ).unsqueeze(0)

#     edge_mask = edge_mask.expand(
#         B,
#         N,
#         N
#     )

#     edge_nll = 0.5 * (
#         pair_residual / edge_var_pair
#         + torch.log(edge_var_pair)
#     )

#     loss_edge_nll = edge_nll[
#         edge_mask
#     ].mean()

#     # --------------------------------------------------
#     # 6. Edge variance monitoring
#     # --------------------------------------------------
#     valid_edge_var = total_var_edge[
#         :,
#         ~torch.eye(
#             N,
#             dtype=torch.bool,
#             device=total_var_edge.device
#         ),
#         :
#     ]

#     edge_var = valid_edge_var.mean()

#     # --------------------------------------------------
#     # 7. Mean prediction variance
#     # --------------------------------------------------
#     pred_mean = y_pred.mean(dim=3)
#     pred_var_mean = pred_mean.var(
#         dim=2,
#         unbiased=False
#     ).mean()

#     # --------------------------------------------------
#     # 8. Best prediction variance
#     # --------------------------------------------------
#     best_decoder_idx_expand = best_decoder_idx[:, :, :, None, None, None].expand(
#         B, N, S, 1, T, 2
#     )

#     best_pred_per_sample = torch.gather(
#         y_pred,
#         dim=3,
#         index=best_decoder_idx_expand
#     ).squeeze(3)

#     pred_var_best = best_pred_per_sample.var(
#         dim=2,
#         unbiased=False
#     ).mean()

#     # --------------------------------------------------
#     # 9. Total loss
#     # --------------------------------------------------
#     loss_nll = lambda_nll * loss_edge_nll

#     total_loss = (
#         loss_pred
#         + best_weight * loss_best
#         + loss_nll
#     )

#     loss_dict = {
#         "loss_pred": loss_pred.detach(),
#         "loss_best": loss_best.detach(),
#         "loss_edge_nll": loss_edge_nll.detach(),
#         "loss_nll": loss_nll.detach(),
#         "best_weight": best_weight,
#         "edge_var": edge_var.detach(),
#         "pair_residual": pair_residual[edge_mask].detach().mean(),
#         "pred_var_mean": pred_var_mean.detach(),
#         "pred_var_best": pred_var_best.detach(),
#         "best_sample_ade": best_sample_ade.detach().mean()
#     }

#     return total_loss, loss_dict
# def compute_loss(
#     y_pred,
#     target,
#     edge_uncertainty,
#     lambda_calib=0.01,
#     lambda_mu=0.01,
#     temperature=0.5,
#     min_var=1e-4,
#     max_var=1e3
# ):
#     if y_pred.dim() != 6:
#         raise ValueError(f"y_pred must have shape [B, N, S, K, T, 2], but received {tuple(y_pred.shape)}.")
#     if target.dim() == 5 and target.size(2) == 1:
#         target = target.squeeze(2)
#     if target.dim() != 4:
#         raise ValueError(f"target must have shape [B, N, T, 2] or [B, N, 1, T, 2], but received {tuple(target.shape)}.")
#     required_keys = ["sampled_edges", "mu_edge", "total_var_edge"]
#     for key in required_keys:
#         if key not in edge_uncertainty:
#             raise KeyError(f"edge_uncertainty must contain '{key}'.")
#     sampled_edges = edge_uncertainty["sampled_edges"]
#     mu_edge = edge_uncertainty["mu_edge"]
#     total_var_edge = edge_uncertainty["total_var_edge"]
#     if sampled_edges.dim() != 5:
#         raise ValueError(f"sampled_edges must have shape [B, N, N, S, D], but received {tuple(sampled_edges.shape)}.")
#     if mu_edge.dim() != 4:
#         raise ValueError(f"mu_edge must have shape [B, N, N, D], but received {tuple(mu_edge.shape)}.")
#     if total_var_edge.dim() != 4:
#         raise ValueError(f"total_var_edge must have shape [B, N, N, D], but received {tuple(total_var_edge.shape)}.")

#     B, N, S, K, T, C = y_pred.shape
#     target_expand = target.unsqueeze(2).unsqueeze(3)

#     displacement_error = torch.norm(y_pred - target_expand, dim=-1)
#     decoder_ade = displacement_error.mean(dim=4)
#     best_decoder_ade_per_sample, best_decoder_idx = decoder_ade.min(dim=3)

#     loss_pred = best_decoder_ade_per_sample.mean()
#     # loss_pred = best_decoder_ade_per_sample.min(dim=2)[0].mean()


#     agent_residual = best_decoder_ade_per_sample.mean(dim=2)
#     residual_target = agent_residual.detach().pow(2)

#     edge_var = total_var_edge.clamp(min=min_var, max=max_var)

#     self_mask = torch.eye(N, dtype=torch.bool, device=edge_var.device).view(1, N, N, 1)
#     valid_edge_mask_float = (~self_mask).to(edge_var.dtype)

#     edge_var_sum = (edge_var * valid_edge_mask_float).sum(dim=(2, 3))
#     edge_count = valid_edge_mask_float.sum(dim=(2, 3)).clamp_min(1.0)
#     agent_edge_var = edge_var_sum / edge_count

#     loss_calib = F.smooth_l1_loss(
#         torch.log1p(agent_edge_var),
#         torch.log1p(residual_target)
#     )

#     sample_weights = torch.softmax(
#         -best_decoder_ade_per_sample.detach() / temperature,
#         dim=2
#     )

#     sample_weights_edge = sample_weights[:, :, None, :, None]

#     soft_target_edge = (
#         sampled_edges * sample_weights_edge
#     ).sum(dim=3)

#     soft_target_edge = soft_target_edge.detach()

#     valid_edge_mask = (~self_mask).expand_as(mu_edge)

#     loss_mu = F.smooth_l1_loss(
#         mu_edge[valid_edge_mask],
#         soft_target_edge[valid_edge_mask]
#     )

#     loss_unc = (
#         lambda_calib * loss_calib
#         + lambda_mu * loss_mu
#     )

#     # total_loss = loss_pred + loss_unc
#     total_loss = loss_pred

#     best_agent_sample_score = best_decoder_ade_per_sample.min(dim=2)[0].mean()
#     sample_error_std = best_decoder_ade_per_sample.std(dim=2, unbiased=False).mean()
#     sample_weight_entropy = -(
#         sample_weights * torch.log(sample_weights + 1e-8)
#     ).sum(dim=2).mean()

#     loss_dict = {
#         "loss_pred": loss_pred.detach(),
#         "loss_unc": loss_unc.detach(),
#         "loss_calib": loss_calib.detach(),
#         "loss_mu": loss_mu.detach(),
#         "agent_residual_mean": agent_residual.detach().mean(),
#         "agent_edge_var_mean": agent_edge_var.detach().mean(),
#         "best_agent_sample_score_mean": best_agent_sample_score.detach(),
#         "sample_error_std_mean": sample_error_std.detach(),
#         "sample_weight_entropy": sample_weight_entropy.detach(),
#         "soft_target_edge_error_mean": (
#             mu_edge - soft_target_edge
#         ).pow(2)[valid_edge_mask].mean().detach()
#     }

#     return total_loss, loss_dict
# def compute_loss(y_pred, target, edge_uncertainty, lambda_unc=0.1, min_var=1e-4, max_var=1e3):
#     if y_pred.dim() != 5:
#         raise ValueError(
#             "y_pred must have shape [B, N, K, T, 2], "
#             f"but received {tuple(y_pred.shape)}."
#         )
#     if target.dim() == 5 and target.size(2) == 1:
#         target = target.squeeze(2)
#     if target.dim() != 4:
#         raise ValueError(
#             "target must have shape [B, N, T, 2] or [B, N, 1, T, 2], "
#             f"but received {tuple(target.shape)}."
#         )
#     if isinstance(edge_uncertainty, dict):
#         if "total_var_edge" not in edge_uncertainty:
#             raise KeyError(
#                 "edge_uncertainty must contain 'total_var_edge'."
#             )
#         total_var_edge = edge_uncertainty["total_var_edge"]
#     else:
#         total_var_edge = edge_uncertainty
#     if total_var_edge.dim() not in (3, 4):
#         raise ValueError(
#             "total_var_edge must have shape [B, N, N] or [B, N, N, D], "
#             f"but received {tuple(total_var_edge.shape)}."
#         )
#     target_expand = target.unsqueeze(2)
#     displacement_error = torch.norm(
#         y_pred - target_expand,
#         dim=-1
#     )
#     decoder_ade = displacement_error.mean(
#         dim=3
#     )
#     best_ade = decoder_ade.min(
#         dim=2
#     )[0]
#     loss_pred = best_ade.mean()
#     agent_residual = best_ade.detach().pow(2)
#     pair_residual = 0.5 * (
#         agent_residual.unsqueeze(2)
#         + agent_residual.unsqueeze(1)
#     )
#     if total_var_edge.dim() == 4:
#         edge_var = total_var_edge.mean(
#             dim=-1
#         )
#     else:
#         edge_var = total_var_edge
#     edge_var = edge_var.clamp(
#         min=min_var,
#         max=max_var
#     )
#     if edge_var.shape != pair_residual.shape:
#         raise ValueError(
#             "The reduced edge uncertainty must have shape [B, N, N]. "
#             f"Received edge_var={tuple(edge_var.shape)} and "
#             f"pair_residual={tuple(pair_residual.shape)}."
#         )
#     num_agents = edge_var.size(1)
#     edge_mask = ~torch.eye(
#         num_agents,
#         dtype=torch.bool,
#         device=edge_var.device
#     ).unsqueeze(0)
#     edge_mask = edge_mask.expand_as(
#         edge_var
#     )
#     edge_nll = 0.5 * (
#         pair_residual / edge_var
#         + torch.log(edge_var)
#     )
#     loss_edge_unc = edge_nll[
#         edge_mask
#     ].mean()
#     loss_unc = lambda_unc * loss_edge_unc
#     total_loss = loss_pred + loss_unc
#     loss_dict = {
#         "loss_pred": loss_pred.detach(),
#         "loss_unc": loss_unc.detach(),
#         "loss_edge_unc": loss_edge_unc.detach(),
#         "edge_var_mean": edge_var[edge_mask].detach().mean(),
#         "pair_residual_mean": pair_residual[edge_mask].detach().mean()
#     }
#     return total_loss, loss_dict



def main():
    best_edge_bank = {}
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)

    print('[INFO] The seed is:', seed)
    if not args.test:
        dataset_train = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='train')
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=8, drop_last=True)
        
    dataset_test = NBADataset(obs_len=opts.past_length, pred_len=opts.future_length, mode='test' if args.test else 'val')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=8)

    model = MART(opts).cuda()
    print(model)
    print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-12)

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.decay_step, gamma=opts.decay_gamma)
    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opts.milestones, gamma=opts.decay_gamma)

    model_save_dir = os.path.join('./checkpoints', os.path.basename(args.config).split('.')[0])
    os.makedirs(model_save_dir, exist_ok=True)

    if args.test:
        model_name = args.dataset + '_ckpt_best_uncertain_mse_warmup_nll.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)

        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)

        test_results = test(model_ckpt['epoch'], model, loader_test)

        print('[INFO] Test Best Sample: {} | ADE: {:.5f} | FDE: {:.5f}'.format(
            test_results["best_sample_index"] + 1,
            test_results["best_sample_ade"],
            test_results["best_sample_fde"]
        ))

        exit()


    results = {'epochs': [], 'losses': []}

    best_val_loss = float("inf")
    best_ade = float("inf")
    best_epoch = -1
    best_sample_index = -1

    print('[INFO] The seed is:', seed)


    for epoch in range(0, opts.num_epochs):

        train(epoch, model, optimizer, loader_train, best_edge_bank)

        test_results = test(epoch, model, loader_test)

        current_fde = test_results["best_sample_fde"]
        current_ade = test_results["best_sample_ade"]
        current_sample_index = test_results["best_sample_index"]

        results['epochs'].append(epoch)
        results['losses'].append(current_fde)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_sample_index': current_sample_index,
            'best_sample_ade': current_ade,
            'best_sample_fde': current_fde,
            'sample_ade_4': test_results["ade_4_per_sample"],
            'sample_fde_4': test_results["fde_4_per_sample"]
        }

        if current_fde < best_val_loss:
            best_val_loss = current_fde
            best_ade = current_ade
            best_epoch = epoch
            best_sample_index = current_sample_index

            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best_uncertain_mse_warmup_nll.pth')
            torch.save(state, file_path)

        print('[INFO] Best {} FDE: {:.5f} | ADE: {:.5f} | Edge Sample: {} | Best Epoch: {}\n'.format(
            loader_test.dataset.mode.capitalize(),
            best_val_loss,
            best_ade,
            best_sample_index + 1,
            best_epoch
        ))

        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_uncertain_mse_warmup_nll_' + str(epoch) + '.pth')

        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_uncertain_mse_warmup_nll_' + str(epoch - 1) + '.pth')
            if os.path.exists(remove_file_path):
                os.remove(remove_file_path)

        torch.save(state, file_path)

        if opts.scheduler_type is not None:
            scheduler.step()
def train(epoch, model, optimizer, loader, best_edge_bank):
    model.train()

    avg_meter = {
        "epoch": epoch,
        "loss": 0.0,
        "loss_pred": 0.0,
        "loss_best": 0.0,
        "loss_edge_nll": 0.0,
        "loss_nll": 0.0,
        "edge_var": 0.0,
        "pair_residual": 0.0,
        "pred_var_mean": 0.0,
        "pred_var_best": 0.0,
        "best_sample_ade": 0.0,
        "counter": 0
    }

    loader_len = len(loader)

    for i, (data, sample_ids) in enumerate(loader):
        optimizer.zero_grad()

        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()

        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]

        y_pred, aux = model(x_abs, x_rel, return_aux=True)

        if opts.pred_rel:
            cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
            y_pred = torch.cumsum(y_pred, dim=4) + cur_pos

        # total_loss, loss_dict = compute_loss(
        #     y_pred=y_pred,
        #     target=y,
        #     edge_uncertainty=aux["edge_uncertainty"],
        #     epoch=epoch,
        #     best_start_epoch=20,
        #     best_warmup_epochs=20,
        #     lambda_best=0.1,
        #     lambda_nll=0.01,
        #     min_var=1e-4,
        #     max_var=1e3
        # )
        total_loss, loss_dict = compute_loss(
            y_pred=y_pred,
            target=y,
            edge_uncertainty=aux["edge_uncertainty"],
            sample_ids=sample_ids,
            best_edge_bank=best_edge_bank,
            epoch=epoch,
            best_start_epoch=20,
            best_warmup_epochs=20,
            lambda_best=0.1,
            lambda_nll=0.01,
            min_var=1e-4,
            max_var=1e3
        )

        meter_weight = batch_size * num_agents

        avg_meter["loss"] += total_loss.item() * meter_weight
        avg_meter["loss_pred"] += loss_dict["loss_pred"].item() * meter_weight
        avg_meter["loss_best"] += loss_dict["loss_best"].item() * meter_weight
        avg_meter["loss_edge_nll"] += loss_dict["loss_edge_nll"].item() * meter_weight
        avg_meter["loss_nll"] += loss_dict["loss_nll"].item() * meter_weight
        avg_meter["edge_var"] += loss_dict["edge_var"].item() * meter_weight
        avg_meter["pair_residual"] += loss_dict["pair_residual"].item() * meter_weight
        avg_meter["pred_var_mean"] += loss_dict["pred_var_mean"].item() * meter_weight
        avg_meter["pred_var_best"] += loss_dict["pred_var_best"].item() * meter_weight
        avg_meter["best_sample_ade"] += loss_dict["best_sample_ade"].item() * meter_weight
        avg_meter["counter"] += meter_weight

        total_loss.backward()

        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)

        optimizer.step()

        if i % 100 == 0:
            th = get_th(opts, model)

            print(
                "[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | "
                "Loss: {:.6f} | Pred: {:.6f} | BestMSE: {:.6f} | BestW: {:.4f} | "
                "EdgeNLL: {:.6f} | NLL: {:.6f} | EdgeVar: {:.6f} | PairResidual: {:.6f} | "
                "PredVarMean: {:.6f} | PredVarBest: {:.6f} | BestSampleADE: {:.6f} | "
                "Threshold: {} | LR: {}"
                .format(
                    loader.dataset.mode.upper(),
                    epoch + 1,
                    opts.num_epochs,
                    i + 1,
                    loader_len,
                    total_loss.item(),
                    loss_dict["loss_pred"].item(),
                    loss_dict["loss_best"].item(),
                    loss_dict["best_weight"],
                    loss_dict["loss_edge_nll"].item(),
                    loss_dict["loss_nll"].item(),
                    loss_dict["edge_var"].item(),
                    loss_dict["pair_residual"].item(),
                    loss_dict["pred_var_mean"].item(),
                    loss_dict["pred_var_best"].item(),
                    loss_dict["best_sample_ade"].item(),
                    th,
                    optimizer.param_groups[0]["lr"]
                )
            )

    return avg_meter["loss"] / avg_meter["counter"]

def test(epoch, model, loader):
    model.eval()
    S = model.num_edge_samples

    avg_meter = {
        "epoch": epoch,
        "ade_1": np.zeros(S), "fde_1": np.zeros(S),
        "ade_2": np.zeros(S), "fde_2": np.zeros(S),
        "ade_3": np.zeros(S), "fde_3": np.zeros(S),
        "ade_4": np.zeros(S), "fde_4": np.zeros(S),
        "oracle_ade_1": 0., "oracle_fde_1": 0.,
        "oracle_ade_2": 0., "oracle_fde_2": 0.,
        "oracle_ade_3": 0., "oracle_fde_3": 0.,
        "oracle_ade_4": 0., "oracle_fde_4": 0.,
        "counter": 0
    }

    with torch.no_grad():
        for _, (data, sample_ids) in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()
            B, N, _, _ = x_abs.shape

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            y_pred, aux = model(x_abs, x_rel, return_aux=True)

            if y_pred.dim() != 6:
                raise ValueError(f"Expected [B,N,S,K,T,2], got {tuple(y_pred.shape)}")
            if y_pred.size(2) != S:
                raise ValueError(f"Expected {S} edge samples, got {y_pred.size(2)}")
            if y_pred.size(3) != 20:
                raise ValueError(f"Expected 20 decoders, got {y_pred.size(3)}")

            if opts.pred_rel:
                cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
                y_pred = torch.cumsum(y_pred, dim=4) + cur_pos

            y_pred = y_pred.cpu().numpy()                     # [B,N,S,K,T,2]
            target = y.cpu().numpy()[:, :, None, :, :]        # [B,N,1,T,2]
            w = B * N

            # --------------------------------------------------
            # Per-sample strict Best-of-20
            # --------------------------------------------------
            for s in range(S):
                pred = y_pred[:, :, s]                        # [B,N,K,T,2]

                for key, end in [("1", 5), ("2", 10), ("3", 15), ("4", pred.shape[3])]:
                    dist = np.linalg.norm(pred[:, :, :, :end] - target[:, :, :, :end], axis=-1)
                    ade = np.mean(np.min(np.mean(dist, axis=3), axis=2))
                    fde = np.mean(np.min(dist[:, :, :, -1], axis=2))

                    avg_meter[f"ade_{key}"][s] += ade * w
                    avg_meter[f"fde_{key}"][s] += fde * w

            # --------------------------------------------------
            # Oracle across S × K
            # --------------------------------------------------
            target_all = target[:, :, None]                   # [B,N,1,1,T,2]
            dist_all = np.linalg.norm(y_pred - target_all, axis=-1)  # [B,N,S,K,T]

            for key, end in [("1", 5), ("2", 10), ("3", 15), ("4", dist_all.shape[-1])]:
                ade_all = np.mean(dist_all[:, :, :, :, :end], axis=4).reshape(B, N, -1)
                fde_all = dist_all[:, :, :, :, end - 1].reshape(B, N, -1)

                avg_meter[f"oracle_ade_{key}"] += np.mean(np.min(ade_all, axis=2)) * w
                avg_meter[f"oracle_fde_{key}"] += np.mean(np.min(fde_all, axis=2)) * w

            avg_meter["counter"] += w

    c = avg_meter["counter"]

    ade_1 = avg_meter["ade_1"] / c
    fde_1 = avg_meter["fde_1"] / c
    ade_2 = avg_meter["ade_2"] / c
    fde_2 = avg_meter["fde_2"] / c
    ade_3 = avg_meter["ade_3"] / c
    fde_3 = avg_meter["fde_3"] / c
    ade_4 = avg_meter["ade_4"] / c
    fde_4 = avg_meter["fde_4"] / c

    oracle_ade_1 = avg_meter["oracle_ade_1"] / c
    oracle_fde_1 = avg_meter["oracle_fde_1"] / c
    oracle_ade_2 = avg_meter["oracle_ade_2"] / c
    oracle_fde_2 = avg_meter["oracle_fde_2"] / c
    oracle_ade_3 = avg_meter["oracle_ade_3"] / c
    oracle_fde_3 = avg_meter["oracle_fde_3"] / c
    oracle_ade_4 = avg_meter["oracle_ade_4"] / c
    oracle_fde_4 = avg_meter["oracle_fde_4"] / c

    th = get_th(opts, model)
    print(f"\n[{loader.dataset.mode.upper()}] Epoch {epoch} th: {th}")

    for s in range(S):
        print("=" * 60)
        print(f"[{loader.dataset.mode.upper()}] Sample {s + 1}")
        print(f"1.0s: {ade_1[s]:.3f}/{fde_1[s]:.3f}")
        print(f"2.0s: {ade_2[s]:.3f}/{fde_2[s]:.3f}")
        print(f"3.0s: {ade_3[s]:.3f}/{fde_3[s]:.3f}")
        print(f"4.0s: {ade_4[s]:.3f}/{fde_4[s]:.3f}")

    best_idx = int(np.argmin(fde_4))
    best_ade, best_fde = float(ade_4[best_idx]), float(fde_4[best_idx])

    print("=" * 60)
    print(f"[BEST FIXED SAMPLE] Sample {best_idx + 1} | 4.0s: {best_ade:.3f}/{best_fde:.3f}")

    print("=" * 60)
    print(f"[ORACLE ALL {S}×20]")
    print(f"1.0s: {oracle_ade_1:.3f}/{oracle_fde_1:.3f}")
    print(f"2.0s: {oracle_ade_2:.3f}/{oracle_fde_2:.3f}")
    print(f"3.0s: {oracle_ade_3:.3f}/{oracle_fde_3:.3f}")
    print(f"4.0s: {oracle_ade_4:.3f}/{oracle_fde_4:.3f}")

    oracle_ade_gain = best_ade - oracle_ade_4
    oracle_fde_gain = best_fde - oracle_fde_4
    print(f"[ORACLE GAIN] ADE: {oracle_ade_gain:.4f} | FDE: {oracle_fde_gain:.4f}\n")

    return {
        "ade_1_per_sample": ade_1, "fde_1_per_sample": fde_1,
        "ade_2_per_sample": ade_2, "fde_2_per_sample": fde_2,
        "ade_3_per_sample": ade_3, "fde_3_per_sample": fde_3,
        "ade_4_per_sample": ade_4, "fde_4_per_sample": fde_4,
        "best_sample_index": best_idx,
        "best_sample_ade": best_ade,
        "best_sample_fde": best_fde,
        "oracle_ade_1": oracle_ade_1, "oracle_fde_1": oracle_fde_1,
        "oracle_ade_2": oracle_ade_2, "oracle_fde_2": oracle_fde_2,
        "oracle_ade_3": oracle_ade_3, "oracle_fde_3": oracle_fde_3,
        "oracle_ade_4": oracle_ade_4, "oracle_fde_4": oracle_fde_4,
        "oracle_ade_gain": oracle_ade_gain,
        "oracle_fde_gain": oracle_fde_gain
    }
# def test(epoch, model, loader):
#     model.eval()
#     avg_meter = {'epoch': epoch, 'ade_1': 0, 'ade_2': 0, 'ade_3': 0, 'ade_4': 0, 'fde_1': 0, 'fde_2': 0, 'fde_3': 0, 'fde_4': 0, 'counter': 0}
    
#     with torch.no_grad():
#         for _, data in enumerate(loader):
#             x_abs, y = data
#             x_abs, y = x_abs.cuda(), y.cuda()        
            
#             batch_size, num_agents, length, _ = x_abs.size()

#             x_rel = torch.zeros_like(x_abs)
#             x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
#             x_rel[:, :, 0] = x_rel[:, :, 1]
            
#             y_pred,aux = model(x_abs, x_rel, return_aux=True)

#             y_pred = y_pred.reshape(batch_size, num_agents, opts.sample_k*1, opts.future_length, 2)

#             if opts.pred_rel:
#                 cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
#                 y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

#             y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
#             y = np.array(y.cpu()) # B, N, T, 2
#             y = y[:, :, None, :, :]
            
#             ade_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :5] - y[:, :, :, :5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             fde_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 4:5] - y[:, :, :, 4:5], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             ade_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :10] - y[:, :, :, :10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             fde_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 9:10] - y[:, :, :, 9:10], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             ade_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :15] - y[:, :, :, :15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             fde_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 14:15] - y[:, :, :, 14:15], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             ade_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
#             fde_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
#             avg_meter['ade_1'] += ade_1
#             avg_meter['fde_1'] += fde_1
#             avg_meter['ade_2'] += ade_2
#             avg_meter['fde_2'] += fde_2
#             avg_meter['ade_3'] += ade_3
#             avg_meter['fde_3'] += fde_3
#             avg_meter['ade_4'] += ade_4
#             avg_meter['fde_4'] += fde_4
            
#             avg_meter['counter'] += (num_agents * batch_size)
    
#     th = get_th(opts, model)
#     print('\n[{}] Epoch {} th: {}'.format(loader.dataset.mode.upper(), epoch, th))
#     print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
#     print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
#     print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
#     print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))
    
#     return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='nba', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--lambda_unc',type=float,default=0.1)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()