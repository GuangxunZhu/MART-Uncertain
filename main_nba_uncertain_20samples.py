import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import UNCERTAINMARTREASONING as MART
from loaders.dataloader_nba import NBADataset
import torch
import torch.nn.functional as F
def compute_loss(
    y_pred,
    target,
    edge_uncertainty,
    lambda_nll=0.1,
    min_var=1e-4,
    max_var=1e3
):
    if target.dim() == 5 and target.size(2) == 1:
        target = target.squeeze(2)

    if y_pred.dim() != 6:
        raise ValueError(
            f"Expected y_pred [B,N,S,K,T,2], got {tuple(y_pred.shape)}"
        )

    B, N, S, K, T, _ = y_pred.shape

    # ============================================================
    # 1. Prediction loss: Best-of-S
    # ============================================================

    target_expand = target.unsqueeze(2).unsqueeze(3)
    displacement_error = torch.norm(
        y_pred - target_expand,
        dim=-1
    )                                           # [B,N,S,K,T]

    decoder_ade = displacement_error.mean(
        dim=4
    )                                           # [B,N,S,K]

    best_decoder_ade_per_sample, best_decoder_idx = decoder_ade.min(
        dim=3
    )                                           # [B,N,S]

    best_sample_ade, best_sample_idx = best_decoder_ade_per_sample.min(
        dim=2
    )                                           # [B,N]

    loss_pred = best_sample_ade.mean()

    # ============================================================
    # 2. Edge uncertainty
    # ============================================================

    mu_edge = edge_uncertainty[
        "mu_edge"
    ]                                           # [B,N,N,D]

    total_var_edge = edge_uncertainty[
        "total_var_edge"
    ]                                           # [B,N,N,D]

    # ============================================================
    # 3. Trajectory residual -> Pair residual
    #
    # detach:
    # NLL 用 residual 作为 uncertainty calibration target，
    # 不反向修改 prediction branch
    # ============================================================

    agent_residual = best_sample_ade.detach().pow(
        2
    )                                           # [B,N]
    # # K=1 current setting
    # mean_pred = y_pred.mean(dim=2).mean(dim=2)
    # # [B,N,T,2]

    # mean_ade = torch.norm(
    #     mean_pred - target,
    #     dim=-1
    # ).mean(dim=-1)
    # # [B,N]

    # agent_residual = mean_ade.detach().pow(2)

    # pair_residual = 0.5 * (
    #     agent_residual.unsqueeze(2)
    #     + agent_residual.unsqueeze(1)
    # )

    pair_residual = 0.5 * (
        agent_residual.unsqueeze(2)
        + agent_residual.unsqueeze(1)
    )                                           # [B,N,N]

    # ============================================================
    # 4. Edge variance
    # ============================================================

    edge_var_pair = total_var_edge.mean(
        dim=-1
    ).clamp(
        min=min_var,
        max=max_var
    )                                           # [B,N,N]

    edge_mask = ~torch.eye(
        N,
        dtype=torch.bool,
        device=mu_edge.device
    ).unsqueeze(0)

    edge_mask = edge_mask.expand(
        B,
        N,
        N
    )

    # ============================================================
    # 5. Gaussian NLL
    #
    # 0.5 * (r / var + log(var))
    # ============================================================

    edge_nll = 0.5 * (
        pair_residual / edge_var_pair
        + torch.log(edge_var_pair)
    )

    loss_edge_nll = edge_nll[
        edge_mask
    ].mean()

    loss_nll = lambda_nll * loss_edge_nll

    # ============================================================
    # 6. Monitoring: variance
    # ============================================================

    edge_var = edge_var_pair[
        edge_mask
    ].mean()

    # ============================================================
    # 7. Monitoring: trajectory variance across S
    #
    # 每个 interaction sample 先选自己的 best decoder
    # ============================================================

    best_decoder_idx_expand = best_decoder_idx[
        :, :, :, None, None, None
    ].expand(
        B,
        N,
        S,
        1,
        T,
        2
    )

    best_pred_per_sample = torch.gather(
        y_pred,
        dim=3,
        index=best_decoder_idx_expand
    ).squeeze(
        3
    )                                           # [B,N,S,T,2]

    pred_var = best_pred_per_sample.var(
        dim=2,
        unbiased=False
    ).mean()

    # ============================================================
    # 8. Monitoring: sample diversity / Best-of-S gain
    # ============================================================

    sample_ade_mean = best_decoder_ade_per_sample.mean()

    best_sample_gain = (
        sample_ade_mean
        - best_sample_ade.mean()
    )

    # ============================================================
    # 9. Extra uncertainty statistics
    # ============================================================

    U_e_edge = edge_uncertainty[
        "U_e_edge"
    ]

    U_a_edge = edge_uncertainty[
        "U_a_edge"
    ]

    epistemic_var = U_e_edge.mean(
        dim=-1
    )[edge_mask].mean()

    aleatoric_var = U_a_edge.mean(
        dim=-1
    )[edge_mask].mean()

    # ============================================================
    # 10. Total loss
    # ============================================================

    total_loss = (
        loss_pred
        + loss_nll
    )

    loss_dict = {
        "loss_pred": loss_pred.detach(),
        "loss_edge_nll": loss_edge_nll.detach(),
        "loss_nll": loss_nll.detach(),

        "edge_var": edge_var.detach(),
        "epistemic_var": epistemic_var.detach(),
        "aleatoric_var": aleatoric_var.detach(),

        "pair_residual": pair_residual[
            edge_mask
        ].mean().detach(),

        "pred_var": pred_var.detach(),

        "sample_ade_mean": sample_ade_mean.detach(),
        "best_sample_ade": best_sample_ade.mean().detach(),
        "best_sample_gain": best_sample_gain.detach()
    }

    return total_loss, loss_dict

def train(epoch, model, optimizer, loader):
    model.train()

    # ==================================================
    # [MODIFIED]
    # 删除 loss_best / best_weight
    # 增加 epistemic_var / aleatoric_var
    # ==================================================
    avg_meter = {
        "epoch": epoch,
        "loss": 0.0,
        "loss_pred": 0.0,
        "loss_edge_nll": 0.0,
        "loss_nll": 0.0,
        "edge_var": 0.0,
        "epistemic_var": 0.0,
        "aleatoric_var": 0.0,
        "pair_residual": 0.0,
        "pred_var": 0.0,
        "sample_ade_mean": 0.0,
        "best_sample_ade": 0.0,
        "best_sample_gain": 0.0,
        "counter": 0
    }

    loader_len = len(loader)

    for i, (data, sample_ids) in enumerate(loader):
        optimizer.zero_grad()

        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()

        batch_size, num_agents, _, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]

        # ==================================================
        # Model output:
        # y_pred [B,N,S,K,T,2]
        # ==================================================
        y_pred, aux = model(
            x_abs,
            x_rel,
            return_aux=True
        )

        if opts.pred_rel:
            cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
            y_pred = torch.cumsum(
                y_pred,
                dim=4
            ) + cur_pos

        # ==================================================
        # [MODIFIED]
        # 当前 loss:
        # L = L_pred + lambda_nll * L_edge_nll
        # ==================================================
        total_loss, loss_dict = compute_loss(
            y_pred=y_pred,
            target=y,
            edge_uncertainty=aux["edge_uncertainty"],

            # 建议先从 0.01 开始
            lambda_nll=0.01,

            min_var=1e-4,
            max_var=1e3
        )

        total_loss.backward()

        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                opts.clip_grad
            )

        optimizer.step()

        meter_weight = batch_size * num_agents

        avg_meter["loss"] += (
            total_loss.item()
            * meter_weight
        )

        avg_meter["loss_pred"] += (
            loss_dict["loss_pred"].item()
            * meter_weight
        )

        avg_meter["loss_edge_nll"] += (
            loss_dict["loss_edge_nll"].item()
            * meter_weight
        )

        avg_meter["loss_nll"] += (
            loss_dict["loss_nll"].item()
            * meter_weight
        )

        avg_meter["edge_var"] += (
            loss_dict["edge_var"].item()
            * meter_weight
        )

        # ==================================================
        # [NEW]
        # ==================================================
        avg_meter["epistemic_var"] += (
            loss_dict["epistemic_var"].item()
            * meter_weight
        )

        avg_meter["aleatoric_var"] += (
            loss_dict["aleatoric_var"].item()
            * meter_weight
        )

        avg_meter["pair_residual"] += (
            loss_dict["pair_residual"].item()
            * meter_weight
        )

        avg_meter["pred_var"] += (
            loss_dict["pred_var"].item()
            * meter_weight
        )

        avg_meter["sample_ade_mean"] += (
            loss_dict["sample_ade_mean"].item()
            * meter_weight
        )

        avg_meter["best_sample_ade"] += (
            loss_dict["best_sample_ade"].item()
            * meter_weight
        )

        avg_meter["best_sample_gain"] += (
            loss_dict["best_sample_gain"].item()
            * meter_weight
        )

        avg_meter["counter"] += meter_weight

        # ==================================================
        # Logging
        # ==================================================
        if i % 100 == 0:
            th = get_th(
                opts,
                model
            )

            # ==================================================
            # [MODIFIED]
            #
            # 删除：
            # Best
            # Weight
            #
            # 增加：
            # EpiVar
            # AleaVar
            # ==================================================
            print(
                "[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | "
                "Loss: {:.6f} | Pred: {:.6f} | "
                "EdgeNLL: {:.6f} | NLL: {:.6f} | "
                "EdgeVar: {:.6f} | EpiVar: {:.6f} | AleaVar: {:.6f} | "
                "PairResidual: {:.6f} | PredVar: {:.6f} | "
                "SampleADEMean: {:.6f} | BestSampleADE: {:.6f} | Gain: {:.6f} | "
                "Threshold: {} | LR: {}"
                .format(
                    loader.dataset.mode.upper(),
                    epoch + 1,
                    opts.num_epochs,
                    i + 1,
                    loader_len,

                    total_loss.item(),
                    loss_dict["loss_pred"].item(),

                    loss_dict["loss_edge_nll"].item(),
                    loss_dict["loss_nll"].item(),

                    loss_dict["edge_var"].item(),
                    loss_dict["epistemic_var"].item(),
                    loss_dict["aleatoric_var"].item(),

                    loss_dict["pair_residual"].item(),
                    loss_dict["pred_var"].item(),

                    loss_dict["sample_ade_mean"].item(),
                    loss_dict["best_sample_ade"].item(),
                    loss_dict["best_sample_gain"].item(),

                    th,
                    optimizer.param_groups[0]["lr"]
                )
            )

    # ==================================================
    # Epoch average
    # ==================================================
    counter = avg_meter["counter"]

    print(
        "\n[TRAIN AVG] "
        "Loss: {:.6f} | Pred: {:.6f} | "
        "EdgeNLL: {:.6f} | EdgeVar: {:.6f} | "
        "EpiVar: {:.6f} | AleaVar: {:.6f} | "
        "PairResidual: {:.6f} | PredVar: {:.6f} | Gain: {:.6f}\n"
        .format(
            avg_meter["loss"] / counter,
            avg_meter["loss_pred"] / counter,
            avg_meter["loss_edge_nll"] / counter,
            avg_meter["edge_var"] / counter,
            avg_meter["epistemic_var"] / counter,
            avg_meter["aleatoric_var"] / counter,
            avg_meter["pair_residual"] / counter,
            avg_meter["pred_var"] / counter,
            avg_meter["best_sample_gain"] / counter
        )
    )

    return avg_meter["loss"] / counter

def test(epoch, model, loader):
    model.eval()

    S = model.num_edge_samples
    K = model.num_decoders

    avg_meter = {
        "epoch": epoch,

        "ade_1": 0.0,
        "fde_1": 0.0,

        "ade_2": 0.0,
        "fde_2": 0.0,

        "ade_3": 0.0,
        "fde_3": 0.0,

        "ade_4": 0.0,
        "fde_4": 0.0,

        "counter": 0
    }

    with torch.no_grad():
        for _, (data, sample_ids) in enumerate(loader):
            x_abs, y = data

            x_abs = x_abs.cuda()
            y = y.cuda()

            B, N, _, _ = x_abs.shape

            # ==================================================
            # Relative trajectory
            # ==================================================
            x_rel = torch.zeros_like(
                x_abs
            )

            x_rel[:, :, 1:] = (
                x_abs[:, :, 1:]
                - x_abs[:, :, :-1]
            )

            x_rel[:, :, 0] = (
                x_rel[:, :, 1]
            )

            # ==================================================
            # Prediction
            #
            # [B,N,S,K,T,2]
            # ==================================================
            y_pred, aux = model(
                x_abs,
                x_rel,
                return_aux=True
            )

            if y_pred.dim() != 6:
                raise ValueError(
                    f"Expected [B,N,S,K,T,2], got {tuple(y_pred.shape)}"
                )

            if y_pred.size(2) != S:
                raise ValueError(
                    f"Expected {S} edge samples, got {y_pred.size(2)}"
                )

            if y_pred.size(3) != K:
                raise ValueError(
                    f"Expected {K} decoders, got {y_pred.size(3)}"
                )

            # ==================================================
            # Relative -> Absolute
            # ==================================================
            if opts.pred_rel:
                cur_pos = x_abs[
                    :, :, -1:, :2
                ].unsqueeze(
                    2
                ).unsqueeze(
                    3
                )

                y_pred = torch.cumsum(
                    y_pred,
                    dim=4
                ) + cur_pos

            # ==================================================
            # NumPy
            # ==================================================
            y_pred = y_pred.cpu().numpy()
            # [B,N,S,K,T,2]

            target = y.cpu().numpy()[
                :, :, None, None, :, :
            ]
            # [B,N,1,1,T,2]

            w = B * N

            dist = np.linalg.norm(
                y_pred - target,
                axis=-1
            )
            # [B,N,S,K,T]

            # ==================================================
            # Evaluation at different horizons
            # ==================================================
            for key, end in [
                ("1", 5),
                ("2", 10),
                ("3", 15),
                ("4", dist.shape[-1])
            ]:
                ade_all = np.mean(
                    dist[:, :, :, :, :end],
                    axis=4
                )
                # [B,N,S,K]

                fde_all = dist[
                    :, :, :, :, end - 1
                ]
                # [B,N,S,K]

                # ==============================================
                # Best-of-K within every edge sample
                # ==============================================
                best_ade_per_sample = np.min(
                    ade_all,
                    axis=3
                )
                # [B,N,S]

                best_fde_per_sample = np.min(
                    fde_all,
                    axis=3
                )
                # [B,N,S]

                # ==============================================
                # Best-of-S edge samples
                # ==============================================
                best_ade = np.min(
                    best_ade_per_sample,
                    axis=2
                )
                # [B,N]

                best_fde = np.min(
                    best_fde_per_sample,
                    axis=2
                )
                # [B,N]

                avg_meter[
                    f"ade_{key}"
                ] += np.mean(
                    best_ade
                ) * w

                avg_meter[
                    f"fde_{key}"
                ] += np.mean(
                    best_fde
                ) * w

            avg_meter["counter"] += w

    c = avg_meter[
        "counter"
    ]

    ade_1 = avg_meter["ade_1"] / c
    fde_1 = avg_meter["fde_1"] / c

    ade_2 = avg_meter["ade_2"] / c
    fde_2 = avg_meter["fde_2"] / c

    ade_3 = avg_meter["ade_3"] / c
    fde_3 = avg_meter["fde_3"] / c

    ade_4 = avg_meter["ade_4"] / c
    fde_4 = avg_meter["fde_4"] / c

    th = get_th(
        opts,
        model
    )

    print(
        f"\n[{loader.dataset.mode.upper()}] "
        f"Epoch {epoch} th: {th}"
    )

    print("=" * 60)

    # ==================================================
    # 注意：
    # K decoder × S edge samples
    # 实际候选数是 S*K
    # ==================================================
    print(
        f"[BEST-OF-{K} DECODERS -> "
        f"BEST-OF-{S} EDGE SAMPLES]"
    )

    print(
        "[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}".format(
            loader.dataset.mode.upper(),
            ade_1,
            fde_1
        )
    )

    print(
        "[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}".format(
            loader.dataset.mode.upper(),
            ade_2,
            fde_2
        )
    )

    print(
        "[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}".format(
            loader.dataset.mode.upper(),
            ade_3,
            fde_3
        )
    )

    print(
        "[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}".format(
            loader.dataset.mode.upper(),
            ade_4,
            fde_4
        )
    )

    print(
        "=" * 60 + "\n"
    )

    return {
        "ade_1": ade_1,
        "fde_1": fde_1,

        "ade_2": ade_2,
        "fde_2": fde_2,

        "ade_3": ade_3,
        "fde_3": fde_3,

        "ade_4": ade_4,
        "fde_4": fde_4
    }

def main():

    # ==================================================
    # [REMOVED]
    #
    # 原来：
    # best_edge_bank = {}
    #
    # 现在不再需要
    # ==================================================

    if args.seed >= 0:
        seed = args.seed
        setup_seed(
            seed
        )
    else:
        seed = random.randint(
            0,
            1000
        )

        setup_seed(
            seed
        )

    print(
        '[INFO] The seed is:',
        seed
    )

    # ==================================================
    # Dataset
    # ==================================================

    if not args.test:
        dataset_train = NBADataset(
            obs_len=opts.past_length,
            pred_len=opts.future_length,
            mode='train'
        )

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )

    dataset_test = NBADataset(
        obs_len=opts.past_length,
        pred_len=opts.future_length,
        mode='test' if args.test else 'val'
    )

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=8
    )

    # ==================================================
    # Model
    # ==================================================

    model = MART(
        opts
    ).cuda()

    print(
        model
    )

    print(
        '[INFO] Model params: {}'.format(
            sum(
                p.numel()
                for p in model.parameters()
            )
        )
    )

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=opts.lr,
        weight_decay=1e-12
    )

    # ==================================================
    # [MODIFIED]
    # 防止 scheduler_type=None 时 scheduler 未定义
    # ==================================================

    scheduler = None

    if opts.scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=opts.decay_step,
            gamma=opts.decay_gamma
        )

    elif opts.scheduler_type == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opts.milestones,
            gamma=opts.decay_gamma
        )

    # ==================================================
    # Checkpoint directory
    # ==================================================

    model_save_dir = os.path.join(
        './checkpoints',
        os.path.basename(
            args.config
        ).split('.')[0]
    )

    os.makedirs(
        model_save_dir,
        exist_ok=True
    )

    # ==================================================
    # Test only
    # ==================================================

    if args.test:

        # ==================================================
        # [MODIFIED]
        # 删除 history
        # ==================================================

        model_name = (
            args.dataset
            + '_ckpt_best_uncertain_edge_attention_nll.pth'
        )

        model_path = os.path.join(
            model_save_dir,
            model_name
        )

        print(
            '[INFO] Loading model from:',
            model_path
        )

        model_ckpt = torch.load(
            model_path
        )

        model.load_state_dict(
            model_ckpt[
                'state_dict'
            ],
            strict=True
        )

        test_results = test(
            model_ckpt[
                'epoch'
            ],
            model,
            loader_test
        )

        print(
            '[INFO] Test Best-of-{} | '
            'ADE: {:.5f} | FDE: {:.5f}'
            .format(
                model.num_edge_samples,
                test_results[
                    "ade_4"
                ],
                test_results[
                    "fde_4"
                ]
            )
        )

        exit()

    # ==================================================
    # Training
    # ==================================================

    results = {
        'epochs': [],
        'losses': []
    }

    best_val_fde = float(
        "inf"
    )

    best_val_ade = float(
        "inf"
    )

    best_epoch = -1

    for epoch in range(
        opts.num_epochs
    ):

        # ==================================================
        # [MODIFIED]
        #
        # 原来：
        # train(..., best_edge_bank)
        #
        # 现在：
        # ==================================================

        train_loss = train(
            epoch,
            model,
            optimizer,
            loader_train
        )

        test_results = test(
            epoch,
            model,
            loader_test
        )

        current_fde = test_results[
            "fde_4"
        ]

        current_ade = test_results[
            "ade_4"
        ]

        results[
            'epochs'
        ].append(
            epoch
        )

        results[
            'losses'
        ].append(
            current_fde
        )

        # ==================================================
        # Checkpoint state
        # ==================================================

        state = {
            'epoch': epoch,

            'state_dict': model.state_dict(),

            'optimizer': optimizer.state_dict(),

            'train_loss': train_loss,

            'ade_1': test_results["ade_1"],
            'fde_1': test_results["fde_1"],

            'ade_2': test_results["ade_2"],
            'fde_2': test_results["fde_2"],

            'ade_3': test_results["ade_3"],
            'fde_3': test_results["fde_3"],

            'ade_4': test_results["ade_4"],
            'fde_4': test_results["fde_4"]
        }

        # ==================================================
        # Best checkpoint
        #
        # 仍然按照 minFDE@4s 保存
        # ==================================================

        if current_fde < best_val_fde:

            best_val_fde = (
                current_fde
            )

            best_val_ade = (
                current_ade
            )

            best_epoch = (
                epoch
            )

            # ==================================================
            # [MODIFIED]
            # 删除 history
            # ==================================================

            best_path = os.path.join(
                model_save_dir,
                str(args.dataset)
                + '_ckpt_best_uncertain_edge_attention_nll.pth'
            )

            torch.save(
                state,
                best_path
            )

        print(
            '[INFO] Best {} Best-of-{} | '
            'FDE: {:.5f} | ADE: {:.5f} | '
            'Best Epoch: {}\n'
            .format(
                loader_test.dataset.mode.capitalize(),
                model.num_edge_samples,
                best_val_fde,
                best_val_ade,
                best_epoch
            )
        )

        # ==================================================
        # Save current epoch
        # ==================================================

        file_path = os.path.join(
            model_save_dir,
            str(args.dataset)
            + '_ckpt_uncertain_edge_attention_nll_'
            + str(epoch)
            + '.pth'
        )

        if epoch > 0:

            remove_file_path = os.path.join(
                model_save_dir,
                str(args.dataset)
                + '_ckpt_uncertain_edge_attention_nll_'
                + str(epoch - 1)
                + '.pth'
            )

            if os.path.exists(
                remove_file_path
            ):
                os.remove(
                    remove_file_path
                )

        torch.save(
            state,
            file_path
        )

        # ==================================================
        # Scheduler
        # ==================================================

        if scheduler is not None:
            scheduler.step()

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