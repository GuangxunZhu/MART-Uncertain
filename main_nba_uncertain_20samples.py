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
def compute_loss(y_pred, target, aux, lambda_epi=0.01, lambda_selector=0.01):
    if target.dim() == 5 and target.size(2) == 1: target = target.squeeze(2)

    # Main Best-of-K
    pred = y_pred[:, :, 0]
    dist = torch.norm(pred - target.unsqueeze(2), dim=-1)
    ade_k = dist.mean(dim=-1)
    best_ade = ade_k.min(dim=2)[0]
    best_fde = dist[:, :, :, -1].min(dim=2)[0]
    loss_main = best_ade.mean()

    # Epistemic [B,N,E,M,T,2]
    epi_pred = aux["epi_samples"]
    B, N, E, M, T, D = epi_pred.shape

    # 每个 head 对 M 次 MC 取平均预测，再用 GT 选择 best head
    epi_mean = epi_pred.mean(dim=3)
    epi_head_ade = torch.norm(epi_mean - target.unsqueeze(2), dim=-1).mean(dim=-1)
    best_epi_ade, best_epi_idx = epi_head_ade.min(dim=2)

    # 只取 GT-best head
    idx = best_epi_idx[:, :, None, None, None, None].expand(B, N, 1, M, T, D)
    best_epi_samples = torch.gather(epi_pred, 2, idx).squeeze(2)

    # best head 内所有 MC realization 都需要学 GT
    loss_epi = torch.norm(
        best_epi_samples - target.unsqueeze(2),
        dim=-1
    ).mean()

    # Selector 学习预测哪个 head 是 GT-best
    selector_logits = aux["epi_selector_logits"]
    loss_selector = F.cross_entropy(
        selector_logits.reshape(B * N, E),
        best_epi_idx.detach().reshape(B * N)
    )

    selector_acc = (
        selector_logits.argmax(dim=-1) == best_epi_idx
    ).float().mean()

    loss = loss_main + lambda_epi * loss_epi + lambda_selector * loss_selector

    return loss, {
        "loss_main": loss_main.detach(),
        "loss_epi": loss_epi.detach(),
        "loss_selector": loss_selector.detach(),
        "selector_acc": selector_acc.detach(),
        "best_epi_ade": best_epi_ade.mean().detach(),
        "best_ade": best_ade.mean().detach(),
        "best_fde": best_fde.mean().detach(),
        "epi_unc": aux["epi_unc"].mean().detach()
    }

def train(epoch, model, optimizer, loader):
    model.train()
    total, counter = 0.0, 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()
        B, N, _, _ = x_abs.shape

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]

        y_pred, aux = model(x_abs, x_rel, return_aux=True)

        if opts.pred_rel:
            cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
            y_pred = torch.cumsum(y_pred, dim=4) + cur_pos

            epi_cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
            # [B,N,1,1,1,2]

            aux["epi_samples"] = (
                torch.cumsum(aux["epi_samples"], dim=4)
                + epi_cur_pos
            )

        loss, d = compute_loss(
            y_pred,
            y,
            aux,
            lambda_epi=getattr(opts, "lambda_epi", 0.01),
            lambda_selector=getattr(opts, "lambda_selector", 0.01)
        )

        loss.backward()

        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)

        optimizer.step()

        w = B * N
        total += loss.item() * w
        counter += w

        if i % 100 == 0:
            print(
                f"[TRAIN] Epoch {epoch+1}/{opts.num_epochs} | "
                f"It {i+1}/{len(loader)} | "
                f"Loss {loss.item():.4f} | "
                f"Main {d['loss_main'].item():.4f} | "
                f"Epi {d['loss_epi'].item():.4f} | "
                f"Sel {d['loss_selector'].item():.4f} | "
                f"SelAcc {d['selector_acc'].item():.3f} | "
                f"EpiBestADE {d['best_epi_ade'].item():.4f} | "
                f"EpiUnc {d['epi_unc'].item():.4f}"
            )

    return total / counter
def test(epoch, model, loader):
    model.eval()
    K = model.num_decoders

    meter = {
        "ade_1": 0., "fde_1": 0.,
        "ade_2": 0., "fde_2": 0.,
        "ade_3": 0., "fde_3": 0.,
        "ade_4": 0., "fde_4": 0.,
        "epi_ade_1": 0., "epi_fde_1": 0.,
        "epi_ade_2": 0., "epi_fde_2": 0.,
        "epi_ade_3": 0., "epi_fde_3": 0.,
        "epi_ade_4": 0., "epi_fde_4": 0.,
        "counter": 0
    }

    unc_ade_all, unc_fde_all = [], []
    ade_all, fde_all = [], []
    epi_ade_all, epi_fde_all = [], []

    with torch.no_grad():
        for x_abs, y in loader:
            x_abs, y = x_abs.cuda(), y.cuda()
            if y.dim() == 5 and y.size(2) == 1: y = y.squeeze(2)

            B, N, _, _ = x_abs.shape

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            # ==================================================
            # Forward
            # selector 在 model 内部选择 epistemic head
            # ==================================================
            y_pred, aux = model(x_abs, x_rel, return_aux=True)

            # ==================================================
            # Main prediction -> absolute
            # [B,N,1,K,T,2]
            # ==================================================
            if opts.pred_rel:
                cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
                y_pred = torch.cumsum(y_pred, dim=4) + cur_pos

            y_pred = y_pred[:, :, 0]
            # [B,N,K,T,2]

            main_dist = torch.norm(
                y_pred - y.unsqueeze(2),
                dim=-1
            )
            # [B,N,K,T]

            T = main_dist.size(-1)
            w = B * N

            # ==================================================
            # Epistemic branch
            # epi_samples: [B,N,E,M,T,2]
            # ==================================================
            epi_samples = aux["epi_samples"]

            if opts.pred_rel:
                epi_cur_pos = x_abs[:, :, -1:, :2].unsqueeze(2).unsqueeze(3)
                epi_samples = torch.cumsum(epi_samples, dim=4) + epi_cur_pos

            # ==================================================
            # Selector-selected epistemic head
            # ==================================================
            selected_idx = aux["epi_selected_idx"]
            # [B,N]

            M = epi_samples.size(3)

            idx = selected_idx[:, :, None, None, None, None].expand(
                B, N, 1, M, T, 2
            )

            selected_epi_samples = torch.gather(
                epi_samples,
                dim=2,
                index=idx
            ).squeeze(2)
            # [B,N,M,T,2]

            # MC samples 是 epistemic samples，不做 Best-of-M
            # 用 MC mean trajectory 计算 epistemic branch ADE/FDE
            selected_epi_mean = selected_epi_samples.mean(dim=2)
            # [B,N,T,2]

            epi_dist = torch.norm(
                selected_epi_mean - y,
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
                # ---------------- Main Best-of-K ----------------
                main_ade_k = main_dist[:, :, :, :end].mean(dim=3)
                main_fde_k = main_dist[:, :, :, end - 1]

                best_main_ade = main_ade_k.min(dim=2)[0]
                best_main_fde = main_fde_k.min(dim=2)[0]

                meter[f"ade_{key}"] += best_main_ade.mean().item() * w
                meter[f"fde_{key}"] += best_main_fde.mean().item() * w

                # ---------------- Selector Epi ----------------
                epi_ade = epi_dist[:, :, :end].mean(dim=2)
                epi_fde = epi_dist[:, :, end - 1]

                meter[f"epi_ade_{key}"] += epi_ade.mean().item() * w
                meter[f"epi_fde_{key}"] += epi_fde.mean().item() * w

                if key == "4":
                    ade_agent = best_main_ade
                    fde_agent = best_main_fde
                    epi_ade_agent = epi_ade
                    epi_fde_agent = epi_fde

            # ==================================================
            # Uncertainty
            # aux["epi_unc"] 本身已经是 selector-selected head uncertainty
            # ==================================================
            unc_ade_all.append(
                aux["epi_unc"].reshape(-1).cpu().numpy()
            )

            unc_fde_all.append(
                aux["epi_unc_final"].reshape(-1).cpu().numpy()
            )

            ade_all.append(
                ade_agent.reshape(-1).cpu().numpy()
            )

            fde_all.append(
                fde_agent.reshape(-1).cpu().numpy()
            )

            epi_ade_all.append(
                epi_ade_agent.reshape(-1).cpu().numpy()
            )

            epi_fde_all.append(
                epi_fde_agent.reshape(-1).cpu().numpy()
            )

            meter["counter"] += w

    # ==================================================
    # Average metrics
    # ==================================================
    c = meter["counter"]
    results = {
        k: meter[k] / c
        for k in meter
        if k != "counter"
    }

    unc_ade_all = np.concatenate(unc_ade_all)
    unc_fde_all = np.concatenate(unc_fde_all)

    ade_all = np.concatenate(ade_all)
    fde_all = np.concatenate(fde_all)

    epi_ade_all = np.concatenate(epi_ade_all)
    epi_fde_all = np.concatenate(epi_fde_all)

    # ==================================================
    # Uncertainty vs MAIN error
    # ==================================================
    rho_ade, p_ade = spearmanr(
        unc_ade_all,
        ade_all
    )

    rho_fde, p_fde = spearmanr(
        unc_fde_all,
        fde_all
    )

    # ==================================================
    # Uncertainty vs EPISTEMIC branch error
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
    bins_ade = compute_uncertainty_bins(
        unc_ade_all,
        ade_all,
        "MAIN-ADE",
        5
    )

    bins_fde = compute_uncertainty_bins(
        unc_fde_all,
        fde_all,
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
    # Print
    # ==================================================
    print(f"\n{'=' * 70}")
    print(f"[{loader.dataset.mode.upper()}] Epoch {epoch}")

    print(f"\n[MAIN BEST-OF-{K}]")
    for t in range(1, 5):
        print(
            f"{t}.0s: "
            f"{results[f'ade_{t}']:.3f}/"
            f"{results[f'fde_{t}']:.3f}"
        )

    print(
        f"\n[SELECTOR EPISTEMIC DECODER | "
        f"E={model.num_epi_decoders} | "
        f"MC={model.epi_samples}]"
    )

    for t in range(1, 5):
        print(
            f"{t}.0s: "
            f"{results[f'epi_ade_{t}']:.3f}/"
            f"{results[f'epi_fde_{t}']:.3f}"
        )

    print(f"\nMean EpiUnc: {unc_ade_all.mean():.6f}")

    print(f"\n[UNCERTAINTY vs MAIN]")
    print(
        f"Spearman(EpiUncMean, MainADE): "
        f"{rho_ade:.6f} (p={p_ade:.3e})"
    )
    print(
        f"Spearman(EpiUncFinal, MainFDE): "
        f"{rho_fde:.6f} (p={p_fde:.3e})"
    )

    print(f"\n[UNCERTAINTY vs EPISTEMIC BRANCH]")
    print(
        f"Spearman(EpiUncMean, EpiADE): "
        f"{rho_epi_ade:.6f} (p={p_epi_ade:.3e})"
    )
    print(
        f"Spearman(EpiUncFinal, EpiFDE): "
        f"{rho_epi_fde:.6f} (p={p_epi_fde:.3e})"
    )

    print("=" * 70)

    # ==================================================
    # Save results
    # ==================================================
    results["spearman_ade"] = float(rho_ade)
    results["spearman_fde"] = float(rho_fde)

    results["spearman_epi_ade"] = float(rho_epi_ade)
    results["spearman_epi_fde"] = float(rho_epi_fde)

    results["bins_ade"] = bins_ade
    results["bins_fde"] = bins_fde
    results["bins_epi_ade"] = bins_epi_ade
    results["bins_epi_fde"] = bins_epi_fde

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





    