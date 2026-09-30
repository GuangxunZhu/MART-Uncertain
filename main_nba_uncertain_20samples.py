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
from scipy.stats import spearmanr
def compute_loss(
    y_pred,
    target,
    aux,
    lambda_epi=0.01
):
    if target.dim() == 5 and target.size(2) == 1:
        target = target.squeeze(2)

    # ==================================================
    # Main Best-of-K
    # ==================================================
    pred = y_pred[:, :, 0]
    # [B,N,K,T,2]

    dist = torch.norm(
        pred - target.unsqueeze(2),
        dim=-1
    )
    # [B,N,K,T]

    ade_k = dist.mean(dim=-1)
    # [B,N,K]

    best_ade = ade_k.min(dim=2)[0]

    best_fde = dist[
        :, :, :, -1
    ].min(dim=2)[0]

    loss_main = best_ade.mean()

    # ==================================================
    # GT-best epistemic decoder
    # ==================================================
    best_epi_samples = aux[
        "best_epi_samples"
    ]
    # [B,N,M,T,2], already absolute

    loss_epi = torch.norm(
        best_epi_samples
        - target.unsqueeze(2),
        dim=-1
    ).mean()

    # ==================================================
    # Total
    # ==================================================
    loss = (
        loss_main
        + lambda_epi * loss_epi
    )

    return loss, {
        "loss_main": loss_main.detach(),
        "loss_epi": loss_epi.detach(),

        "best_ade": best_ade.mean().detach(),
        "best_fde": best_fde.mean().detach(),

        "best_epi_ade": aux[
            "best_epi_ade"
        ].mean().detach(),

        "epi_unc": aux[
            "epi_unc"
        ].mean().detach()
    }
def train(
    epoch,
    model,
    optimizer,
    loader
):
    model.train()

    total, counter = 0.0, 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x_abs, y = data

        x_abs = x_abs.cuda()
        y = y.cuda()

        if y.dim() == 5 and y.size(2) == 1:
            y = y.squeeze(2)

        B, N, _, _ = x_abs.shape

        # ==================================================
        # Relative input
        # ==================================================
        x_rel = torch.zeros_like(x_abs)

        x_rel[:, :, 1:] = (
            x_abs[:, :, 1:]
            - x_abs[:, :, :-1]
        )

        x_rel[:, :, 0] = x_rel[:, :, 1]

        # ==================================================
        # Forward
        # GT selects best epistemic decoder
        # ==================================================
        y_pred, aux = model(
            x_abs,
            x_rel,
            target=y,
            return_aux=True
        )

        # ==================================================
        # Main prediction relative -> absolute
        # ==================================================
        if opts.pred_rel:
            cur_pos = x_abs[
                :, :, -1:, :2
            ].unsqueeze(2).unsqueeze(3)

            y_pred = (
                torch.cumsum(
                    y_pred,
                    dim=4
                )
                + cur_pos
            )

        # ==================================================
        # Loss
        # ==================================================
        loss, d = compute_loss(
            y_pred,
            y,
            aux,
            lambda_epi=getattr(
                opts,
                "lambda_epi",
                0.01
            )
        )

        loss.backward()

        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                opts.clip_grad
            )

        optimizer.step()

        # ==================================================
        # Logging
        # ==================================================
        w = B * N

        total += loss.item() * w
        counter += w

        if i % 100 == 0:
            print(
                f"[TRAIN] "
                f"Epoch {epoch+1}/{opts.num_epochs} | "
                f"It {i+1}/{len(loader)} | "
                f"Loss {loss.item():.4f} | "
                f"Main {d['loss_main'].item():.4f} | "
                f"Epi {d['loss_epi'].item():.4f} | "
                f"EpiBestADE {d['best_epi_ade'].item():.4f} | "
                f"EpiUnc {d['epi_unc'].item():.4f}"
            )

    return total / counter

def test(
    epoch,
    model,
    loader
):
    model.eval()

    K = model.num_decoders
    E = model.num_epi_decoders

    meter = {
        "ade_1": 0.,
        "fde_1": 0.,
        "ade_2": 0.,
        "fde_2": 0.,
        "ade_3": 0.,
        "fde_3": 0.,
        "ade_4": 0.,
        "fde_4": 0.,

        "epi_ade_1": 0.,
        "epi_fde_1": 0.,
        "epi_ade_2": 0.,
        "epi_fde_2": 0.,
        "epi_ade_3": 0.,
        "epi_fde_3": 0.,
        "epi_ade_4": 0.,
        "epi_fde_4": 0.,

        "counter": 0
    }

    unc_ade_all = []
    unc_fde_all = []

    main_ade_all = []
    main_fde_all = []

    epi_ade_all = []
    epi_fde_all = []

    best_epi_idx_all = []

    with torch.no_grad():

        for x_abs, y in loader:
            x_abs = x_abs.cuda()
            y = y.cuda()

            if y.dim() == 5 and y.size(2) == 1:
                y = y.squeeze(2)

            B, N, _, _ = x_abs.shape

            # ==================================================
            # Relative input
            # ==================================================
            x_rel = torch.zeros_like(x_abs)

            x_rel[:, :, 1:] = (
                x_abs[:, :, 1:]
                - x_abs[:, :, :-1]
            )

            x_rel[:, :, 0] = x_rel[:, :, 1]

            # ==================================================
            # Forward
            # GT directly selects best epistemic decoder
            # ==================================================
            y_pred, aux = model(
                x_abs,
                x_rel,
                target=y,
                return_aux=True
            )

            # ==================================================
            # Main prediction -> absolute
            # ==================================================
            if opts.pred_rel:
                cur_pos = x_abs[
                    :, :, -1:, :2
                ].unsqueeze(2).unsqueeze(3)

                y_pred = (
                    torch.cumsum(
                        y_pred,
                        dim=4
                    )
                    + cur_pos
                )

            y_pred = y_pred[:, :, 0]
            # [B,N,K,T,2]

            main_dist = torch.norm(
                y_pred
                - y.unsqueeze(2),
                dim=-1
            )
            # [B,N,K,T]

            T = main_dist.size(-1)
            w = B * N

            # ==================================================
            # Epistemic prediction
            # already GT-best + MC mean + absolute
            # ==================================================
            best_epi_mean = aux[
                "best_epi_mean"
            ]
            # [B,N,T,2]

            epi_dist = torch.norm(
                best_epi_mean - y,
                dim=-1
            )
            # [B,N,T]

            # ==================================================
            # Main + Epistemic metrics
            # ==================================================
            for key, end in [
                ("1", min(5, T)),
                ("2", min(10, T)),
                ("3", min(15, T)),
                ("4", T)
            ]:

                # ------------------------------------------
                # Main Best-of-K
                # ------------------------------------------
                main_ade_k = main_dist[
                    :, :, :, :end
                ].mean(dim=3)

                main_fde_k = main_dist[
                    :, :, :, end - 1
                ]

                best_main_ade = main_ade_k.min(
                    dim=2
                )[0]

                best_main_fde = main_fde_k.min(
                    dim=2
                )[0]

                meter[
                    f"ade_{key}"
                ] += (
                    best_main_ade.mean().item()
                    * w
                )

                meter[
                    f"fde_{key}"
                ] += (
                    best_main_fde.mean().item()
                    * w
                )

                # ------------------------------------------
                # GT-best epistemic decoder
                # MC mean trajectory
                # ------------------------------------------
                epi_ade = epi_dist[
                    :, :, :end
                ].mean(dim=2)

                epi_fde = epi_dist[
                    :, :, end - 1
                ]

                meter[
                    f"epi_ade_{key}"
                ] += (
                    epi_ade.mean().item()
                    * w
                )

                meter[
                    f"epi_fde_{key}"
                ] += (
                    epi_fde.mean().item()
                    * w
                )

                if key == "4":
                    main_ade_agent = best_main_ade
                    main_fde_agent = best_main_fde

                    epi_ade_agent = epi_ade
                    epi_fde_agent = epi_fde

            # ==================================================
            # Uncertainty
            # GT-best head's MC variance
            # ==================================================
            unc_ade_all.append(
                aux["epi_unc"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            unc_fde_all.append(
                aux["epi_unc_final"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            # ==================================================
            # Errors
            # ==================================================
            main_ade_all.append(
                main_ade_agent
                .reshape(-1)
                .cpu()
                .numpy()
            )

            main_fde_all.append(
                main_fde_agent
                .reshape(-1)
                .cpu()
                .numpy()
            )

            epi_ade_all.append(
                epi_ade_agent
                .reshape(-1)
                .cpu()
                .numpy()
            )

            epi_fde_all.append(
                epi_fde_agent
                .reshape(-1)
                .cpu()
                .numpy()
            )

            best_epi_idx_all.append(
                aux["best_epi_idx"]
                .reshape(-1)
                .cpu()
                .numpy()
            )

            meter["counter"] += w

    # ==================================================
    # Average results
    # ==================================================
    c = meter["counter"]

    results = {
        k: meter[k] / c
        for k in meter
        if k != "counter"
    }

    unc_ade_all = np.concatenate(
        unc_ade_all
    )

    unc_fde_all = np.concatenate(
        unc_fde_all
    )

    main_ade_all = np.concatenate(
        main_ade_all
    )

    main_fde_all = np.concatenate(
        main_fde_all
    )

    epi_ade_all = np.concatenate(
        epi_ade_all
    )

    epi_fde_all = np.concatenate(
        epi_fde_all
    )

    best_epi_idx_all = np.concatenate(
        best_epi_idx_all
    )

    # ==================================================
    # Spearman: uncertainty vs main error
    # ==================================================
    rho_main_ade, p_main_ade = spearmanr(
        unc_ade_all,
        main_ade_all
    )

    rho_main_fde, p_main_fde = spearmanr(
        unc_fde_all,
        main_fde_all
    )

    # ==================================================
    # Spearman: uncertainty vs epistemic error
    # ==================================================
    rho_epi_ade, p_epi_ade = spearmanr(
        unc_ade_all,
        epi_ade_all
    )

    rho_epi_fde, p_epi_fde = spearmanr(
        unc_fde_all,
        epi_fde_all
    )

    # ==================================================
    # Bins
    # ==================================================
    bins_main_ade = compute_uncertainty_bins(
        unc_ade_all,
        main_ade_all,
        "MAIN-ADE",
        5
    )

    bins_main_fde = compute_uncertainty_bins(
        unc_fde_all,
        main_fde_all,
        "MAIN-FDE",
        5
    )

    bins_epi_ade = compute_uncertainty_bins(
        unc_ade_all,
        epi_ade_all,
        "EPI-ADE",
        5
    )

    bins_epi_fde = compute_uncertainty_bins(
        unc_fde_all,
        epi_fde_all,
        "EPI-FDE",
        5
    )

    # ==================================================
    # Head usage
    # ==================================================
    head_counts = np.bincount(
        best_epi_idx_all,
        minlength=E
    )

    head_ratio = (
        head_counts
        / max(
            head_counts.sum(),
            1
        )
    )

    # ==================================================
    # Print
    # ==================================================
    print(
        f"\n{'=' * 70}"
    )

    print(
        f"[{loader.dataset.mode.upper()}] "
        f"Epoch {epoch}"
    )

    # ---------------- Main ----------------
    print(
        f"\n[MAIN BEST-OF-{K}]"
    )

    for t in range(1, 5):
        print(
            f"{t}.0s: "
            f"{results[f'ade_{t}']:.3f}/"
            f"{results[f'fde_{t}']:.3f}"
        )

    # ---------------- Epistemic ----------------
    print(
        f"\n[GT-BEST EPISTEMIC DECODER | "
        f"E={E} | "
        f"MC={model.epi_samples}]"
    )

    for t in range(1, 5):
        print(
            f"{t}.0s: "
            f"{results[f'epi_ade_{t}']:.3f}/"
            f"{results[f'epi_fde_{t}']:.3f}"
        )

    # ---------------- Uncertainty ----------------
    print(
        f"\nMean EpiUnc: "
        f"{unc_ade_all.mean():.6f}"
    )

    print(
        f"Mean EpiFinalUnc: "
        f"{unc_fde_all.mean():.6f}"
    )

    print(
        "\n[UNCERTAINTY vs MAIN ERROR]"
    )

    print(
        f"Spearman(EpiUncMean, MainADE): "
        f"{rho_main_ade:.6f} "
        f"(p={p_main_ade:.3e})"
    )

    print(
        f"Spearman(EpiUncFinal, MainFDE): "
        f"{rho_main_fde:.6f} "
        f"(p={p_main_fde:.3e})"
    )

    print(
        "\n[UNCERTAINTY vs EPISTEMIC ERROR]"
    )

    print(
        f"Spearman(EpiUncMean, EpiADE): "
        f"{rho_epi_ade:.6f} "
        f"(p={p_epi_ade:.3e})"
    )

    print(
        f"Spearman(EpiUncFinal, EpiFDE): "
        f"{rho_epi_fde:.6f} "
        f"(p={p_epi_fde:.3e})"
    )

    # ==================================================
    # Head usage
    # ==================================================
    print(
        "\n[GT-BEST EPISTEMIC HEAD USAGE]"
    )

    for e in range(E):
        print(
            f"Head {e:02d} | "
            f"N={head_counts[e]:6d} | "
            f"Ratio={head_ratio[e]:.4f}"
        )

    print(
        "=" * 70
    )

    # ==================================================
    # Store results
    # ==================================================
    results["spearman_main_ade"] = float(
        rho_main_ade
    )

    results["spearman_main_fde"] = float(
        rho_main_fde
    )

    results["spearman_epi_ade"] = float(
        rho_epi_ade
    )

    results["spearman_epi_fde"] = float(
        rho_epi_fde
    )

    # 保留旧接口
    results["spearman_ade"] = float(
        rho_main_ade
    )

    results["spearman_fde"] = float(
        rho_main_fde
    )

    results["bins_main_ade"] = bins_main_ade
    results["bins_main_fde"] = bins_main_fde

    results["bins_epi_ade"] = bins_epi_ade
    results["bins_epi_fde"] = bins_epi_fde

    results["head_counts"] = head_counts.tolist()
    results["head_ratio"] = head_ratio.tolist()

    return results


def compute_uncertainty_bins(uncertainty, error, name="ADE", num_bins=5):
    uncertainty = np.asarray(uncertainty).reshape(-1)
    error = np.asarray(error).reshape(-1)

    valid = np.isfinite(uncertainty) & np.isfinite(error)
    uncertainty, error = uncertainty[valid], error[valid]

    bins = np.array_split(np.argsort(uncertainty), num_bins)
    results = []

    print(f"\n{'=' * 60}\n{name} Epistemic Uncertainty Bins\n{'=' * 60}")

    for i, idx in enumerate(bins):
        r = {
            "bin": i + 1,
            "num_samples": len(idx),
            "uncertainty": float(uncertainty[idx].mean()),
            "error": float(error[idx].mean())
        }
        results.append(r)
        print(
            f"Bin {i+1}/{num_bins} | "
            f"N {len(idx):6d} | "
            f"Unc {r['uncertainty']:.8f} | "
            f"{name} {r['error']:.6f}"
        )

    print("=" * 60)
    return results

def main():
    # ==================================================
    # Seed
    # ==================================================
    seed = args.seed if args.seed >= 0 else random.randint(0, 1000)
    setup_seed(seed)
    print(f"[INFO] Seed: {seed}")

    # ==================================================
    # Dataset
    # ==================================================
    if not args.test:
        dataset_train = NBADataset(
            obs_len=opts.past_length,
            pred_len=opts.future_length,
            mode="train"
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
        mode="test" if args.test else "val"
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
    model = MART(opts).cuda()

    print(model)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters())}")
    print(
        f"[INFO] Main decoders: {model.num_decoders} | "
        f"Epistemic decoders: {model.num_epi_decoders} | "
        f"MC samples/head: {model.epi_samples}"
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
    # Scheduler
    # ==================================================
    scheduler = None

    if getattr(opts, "scheduler_type", None) == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=opts.decay_step,
            gamma=opts.decay_gamma
        )

    elif getattr(opts, "scheduler_type", None) == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=opts.milestones,
            gamma=opts.decay_gamma
        )

    # ==================================================
    # Checkpoint directory
    # ==================================================
    model_save_dir = os.path.join(
        "./checkpoints",
        os.path.basename(args.config).split(".")[0]
    )
    os.makedirs(model_save_dir, exist_ok=True)

    best_name = f"{args.dataset}_ckpt_best_epistemic_decoder.pth"
    best_path = os.path.join(model_save_dir, best_name)

    # ==================================================
    # Test only
    # ==================================================
    if args.test:
        print(f"[INFO] Loading model from: {best_path}")

        model_ckpt = torch.load(best_path, map_location="cuda")
        model.load_state_dict(model_ckpt["state_dict"], strict=True)

        test_results = test(
            model_ckpt["epoch"],
            model,
            loader_test
        )

        print(
            f"[INFO] Test | "
            f"Main Best-of-{model.num_decoders} | "
            f"EpiHeads={model.num_epi_decoders} | "
            f"MC={model.epi_samples} | "
            f"ADE={test_results['ade_4']:.5f} | "
            f"FDE={test_results['fde_4']:.5f} | "
            f"Spearman(U,ADE)={test_results['spearman_ade']:.5f} | "
            f"Spearman(U,FDE)={test_results['spearman_fde']:.5f}"
        )

        return

    # ==================================================
    # Training
    # ==================================================
    results = {
        "epochs": [],
        "losses": [],
        "ade": [],
        "fde": [],
        "spearman_ade": [],
        "spearman_fde": []
    }

    best_val_fde = float("inf")
    best_val_ade = float("inf")
    best_epoch = -1

    for epoch in range(opts.num_epochs):
        # ----------------------------------------------
        # Train
        # ----------------------------------------------
        train_loss = train(
            epoch,
            model,
            optimizer,
            loader_train
        )

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------
        test_results = test(
            epoch,
            model,
            loader_test
        )

        current_ade = test_results["ade_4"]
        current_fde = test_results["fde_4"]

        results["epochs"].append(epoch)
        results["losses"].append(train_loss)
        results["ade"].append(current_ade)
        results["fde"].append(current_fde)
        results["spearman_ade"].append(test_results["spearman_ade"])
        results["spearman_fde"].append(test_results["spearman_fde"])

        # ----------------------------------------------
        # Checkpoint
        # ----------------------------------------------
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,

            "ade_1": test_results["ade_1"],
            "fde_1": test_results["fde_1"],
            "ade_2": test_results["ade_2"],
            "fde_2": test_results["fde_2"],
            "ade_3": test_results["ade_3"],
            "fde_3": test_results["fde_3"],
            "ade_4": current_ade,
            "fde_4": current_fde,

            "spearman_ade": test_results["spearman_ade"],
            "spearman_fde": test_results["spearman_fde"],

            "num_decoders": model.num_decoders,
            "num_epi_decoders": model.num_epi_decoders,
            "epi_samples": model.epi_samples
        }

        # ----------------------------------------------
        # Best checkpoint: minFDE@4s
        # ----------------------------------------------
        if current_fde < best_val_fde:
            best_val_fde = current_fde
            best_val_ade = current_ade
            best_epoch = epoch

            torch.save(state, best_path)

            print(
                f"[INFO] New best checkpoint | "
                f"Epoch={epoch} | "
                f"ADE={best_val_ade:.5f} | "
                f"FDE={best_val_fde:.5f}"
            )

        # ----------------------------------------------
        # Epoch summary
        # ----------------------------------------------
        print(
            f"[INFO] Epoch {epoch+1}/{opts.num_epochs} | "
            f"Main Best-of-{model.num_decoders} | "
            f"EpiHeads={model.num_epi_decoders} | "
            f"MC={model.epi_samples} | "
            f"ADE={current_ade:.5f} | "
            f"FDE={current_fde:.5f} | "
            f"Spearman(U,ADE)={test_results['spearman_ade']:.5f} | "
            f"Spearman(U,FDE)={test_results['spearman_fde']:.5f}"
        )

        # ----------------------------------------------
        # Save current epoch
        # ----------------------------------------------
        file_path = os.path.join(
            model_save_dir,
            f"{args.dataset}_ckpt_epistemic_decoder_{epoch}.pth"
        )

        if epoch > 0:
            old_path = os.path.join(
                model_save_dir,
                f"{args.dataset}_ckpt_epistemic_decoder_{epoch-1}.pth"
            )
            if os.path.exists(old_path): os.remove(old_path)

        torch.save(state, file_path)

        if scheduler is not None: scheduler.step()

    # ==================================================
    # Final summary
    # ==================================================
    print(
        f"[INFO] Training finished | "
        f"Best epoch={best_epoch} | "
        f"Best ADE={best_val_ade:.5f} | "
        f"Best FDE={best_val_fde:.5f}"
    )

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





    