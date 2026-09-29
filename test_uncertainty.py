import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import MART
from loaders.dataloader_nba import NBADataset


def enable_mc_dropout(model):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def get_original_metrics(y_pred, y):
    dist = torch.norm(y_pred - y.unsqueeze(2), dim=-1)  # [B,N,K,T]
    ade_each_k = dist.mean(dim=3)  # [B,N,K]
    fde_each_k = dist[:, :, :, -1]  # [B,N,K]
    min_ade = ade_each_k.min(dim=2)[0]  # [B,N]
    min_fde = fde_each_k.min(dim=2)[0]  # [B,N]
    return min_ade, min_fde


def get_best_trajectory(y_pred, y):
    dist = torch.norm(y_pred - y.unsqueeze(2), dim=-1)  # [B,N,K,T]
    ade_each_k = dist.mean(dim=3)  # [B,N,K]
    best_idx = ade_each_k.argmin(dim=2)  # [B,N]
    gather_idx = best_idx[:, :, None, None, None].expand(-1, -1, 1, y_pred.size(3), y_pred.size(4))
    best_pred = torch.gather(y_pred, dim=2, index=gather_idx).squeeze(2)  # [B,N,T,2]
    return best_pred


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0

    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = rank
        i = j + 1

    return ranks


def spearman_correlation(x, y):
    x_rank = rankdata(x)
    y_rank = rankdata(y)

    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return np.nan

    return np.corrcoef(x_rank, y_rank)[0, 1]


def main():
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
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)

        print('[INFO] Loading model from:', model_path)

        model_ckpt = torch.load(model_path, map_location='cuda')
        model.load_state_dict(model_ckpt['state_dict'], strict=True)

        result_dir = os.path.join('./results', os.path.basename(args.config).split('.')[0])
        os.makedirs(result_dir, exist_ok=True)

        test_uncertainty(model_ckpt['epoch'], model, loader_test, mc_samples=args.mc_samples, num_bins=args.num_bins, save_dir=result_dir)
        exit()

    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0

    for epoch in range(opts.num_epochs):
        train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_ade = ade
            best_epoch = epoch

            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best.pth')
            torch.save(state, file_path)

        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format(loader_test.dataset.mode.capitalize(), best_val_loss, best_ade, best_epoch))

        file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch) + '.pth')

        if epoch > 0:
            remove_file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_' + str(epoch - 1) + '.pth')
            os.system('rm ' + remove_file_path)

        torch.save(state, file_path)

        if opts.scheduler_type is not None:
            scheduler.step()


def train(epoch, model, optimizer, loader):
    model.train()

    avg_meter = {'epoch': epoch, 'loss': 0, 'counter': 0}
    loader_len = len(loader)

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()

        batch_size, num_agents, _, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]

        y_pred = model(x_abs, x_rel)

        if opts.pred_rel:
            cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

        y_expand = y[:, :, None, :, :]
        total_loss = torch.mean(torch.min(torch.mean(torch.norm(y_pred - y_expand, dim=-1), dim=3), dim=2)[0])

        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += batch_size * num_agents

        total_loss.backward()

        if opts.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)

        optimizer.step()

        if i % 100 == 0:
            th = get_th(opts, model)
            print('[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'.format(loader.dataset.mode.upper(), epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))

    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()

    avg_meter = {'epoch': epoch, 'ade_1': 0, 'fde_1': 0, 'ade_2': 0, 'fde_2': 0, 'ade_3': 0, 'fde_3': 0, 'ade_4': 0, 'fde_4': 0, 'counter': 0}

    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()

            batch_size, num_agents, _, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            y_pred = model(x_abs, x_rel)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = y_pred.cpu().numpy()
            y = y.cpu().numpy()[:, :, None, :, :]

            ade_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :5] - y[:, :, :, :5], axis=-1), axis=3), axis=2)) * num_agents * batch_size
            fde_1 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 4:5] - y[:, :, :, 4:5], axis=-1), axis=3), axis=2)) * num_agents * batch_size

            ade_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :10] - y[:, :, :, :10], axis=-1), axis=3), axis=2)) * num_agents * batch_size
            fde_2 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 9:10] - y[:, :, :, 9:10], axis=-1), axis=3), axis=2)) * num_agents * batch_size

            ade_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, :15] - y[:, :, :, :15], axis=-1), axis=3), axis=2)) * num_agents * batch_size
            fde_3 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, 14:15] - y[:, :, :, 14:15], axis=-1), axis=3), axis=2)) * num_agents * batch_size

            ade_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * num_agents * batch_size
            fde_4 = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * num_agents * batch_size

            avg_meter['ade_1'] += ade_1
            avg_meter['fde_1'] += fde_1
            avg_meter['ade_2'] += ade_2
            avg_meter['fde_2'] += fde_2
            avg_meter['ade_3'] += ade_3
            avg_meter['fde_3'] += fde_3
            avg_meter['ade_4'] += ade_4
            avg_meter['fde_4'] += fde_4
            avg_meter['counter'] += num_agents * batch_size

    th = get_th(opts, model)

    print('\n[{}] Epoch {} th: {}'.format(loader.dataset.mode.upper(), epoch, th))
    print('[{}] minADE/minFDE (1.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_1'] / avg_meter['counter'], avg_meter['fde_1'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (2.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_2'] / avg_meter['counter'], avg_meter['fde_2'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (3.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_3'] / avg_meter['counter'], avg_meter['fde_3'] / avg_meter['counter']))
    print('[{}] minADE/minFDE (4.0s): {:.3f}/{:.3f}'.format(loader.dataset.mode.upper(), avg_meter['ade_4'] / avg_meter['counter'], avg_meter['fde_4'] / avg_meter['counter']))

    return avg_meter['fde_4'] / avg_meter['counter'], avg_meter['ade_4'] / avg_meter['counter']


def test_uncertainty(epoch, model, loader, mc_samples=20, num_bins=5, save_dir='./results'):
    agent_csv_path = os.path.join(save_dir, '{}_uncertainty_agent.csv'.format(args.dataset))
    summary_csv_path = os.path.join(save_dir, '{}_uncertainty_summary.csv'.format(args.dataset))
    ade_bin_csv_path = os.path.join(save_dir, '{}_uncertainty_bins_ADE.csv'.format(args.dataset))
    fde_bin_csv_path = os.path.join(save_dir, '{}_uncertainty_bins_FDE.csv'.format(args.dataset))
    uncertainty_path = os.path.join(save_dir, '{}_epistemic_uncertainty.npz'.format(args.dataset))

    all_ade = []
    all_fde = []
    all_unc_mean = []
    all_unc_final = []
    all_epistemic_unc = []
    all_sample_idx = []
    all_agent_idx = []

    global_sample_index = 0

    print('\n==========================================')
    print('Original Model vs MC-Dropout Uncertainty')
    print('==========================================')
    print('[INFO] Epoch:', epoch)
    print('[INFO] MC samples:', mc_samples)

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()

            batch_size, num_agents, _, _ = x_abs.shape

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            # ====================================================
            # Original model
            # ====================================================

            model.eval()

            y_pred_original = model(x_abs, x_rel)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred_original = torch.cumsum(y_pred_original, dim=3) + cur_pos

            original_ade, original_fde = get_original_metrics(y_pred_original, y)

            # ====================================================
            # MC Dropout
            # ====================================================

            enable_mc_dropout(model)

            mc_best_predictions = []

            for _ in range(mc_samples):
                y_pred_mc = model(x_abs, x_rel)

                if opts.pred_rel:
                    cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                    y_pred_mc = torch.cumsum(y_pred_mc, dim=3) + cur_pos

                best_pred = get_best_trajectory(y_pred_mc, y)
                mc_best_predictions.append(best_pred)

            # ====================================================
            # [M,B,N,T,2]
            # ====================================================

            mc_best_predictions = torch.stack(mc_best_predictions, dim=0)

            # ====================================================
            # Var across MC
            # [B,N,T,2]
            # ====================================================

            epistemic_var = mc_best_predictions.var(dim=0, unbiased=False)

            # ====================================================
            # sqrt(var_x + var_y)
            # [B,N,T]
            # ====================================================

            epistemic_unc = torch.sqrt(torch.clamp(epistemic_var.sum(dim=-1), min=0.0))

            # ====================================================
            # Mean trajectory uncertainty / final uncertainty
            # [B,N]
            # ====================================================

            epistemic_unc_mean = epistemic_unc.mean(dim=-1)
            epistemic_unc_final = epistemic_unc[:, :, -1]

            # ====================================================
            # Save batch
            # ====================================================

            all_ade.append(original_ade.cpu().numpy())
            all_fde.append(original_fde.cpu().numpy())
            all_unc_mean.append(epistemic_unc_mean.cpu().numpy())
            all_unc_final.append(epistemic_unc_final.cpu().numpy())
            all_epistemic_unc.append(epistemic_unc.cpu().numpy())

            for b in range(batch_size):
                for n in range(num_agents):
                    all_sample_idx.append(global_sample_index + b)
                    all_agent_idx.append(n)

            global_sample_index += batch_size

            print('[TEST] Batch {:04d}/{:04d} | ADE {:.4f} | FDE {:.4f} | UncMean {:.6f} | UncFinal {:.6f}'.format(
                batch_idx + 1,
                len(loader),
                original_ade.mean().item(),
                original_fde.mean().item(),
                epistemic_unc_mean.mean().item(),
                epistemic_unc_final.mean().item()
            ))

    # ============================================================
    # Concatenate
    # ============================================================

    all_ade = np.concatenate(all_ade, axis=0)
    all_fde = np.concatenate(all_fde, axis=0)
    all_unc_mean = np.concatenate(all_unc_mean, axis=0)
    all_unc_final = np.concatenate(all_unc_final, axis=0)
    all_epistemic_unc = np.concatenate(all_epistemic_unc, axis=0)

    ade_flat = all_ade.reshape(-1)
    fde_flat = all_fde.reshape(-1)
    unc_mean_flat = all_unc_mean.reshape(-1)
    unc_final_flat = all_unc_final.reshape(-1)

    # ============================================================
    # Spearman correlations
    # ============================================================

    spearman_ade = spearman_correlation(unc_mean_flat, ade_flat)
    spearman_fde = spearman_correlation(unc_final_flat, fde_flat)

    # ============================================================
    # Save per-agent results
    # ============================================================

    with open(agent_csv_path, 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(['sample_index', 'agent_index', 'original_minADE', 'original_minFDE', 'epistemic_unc_mean', 'epistemic_unc_final'])

        for i in range(len(ade_flat)):
            csv_writer.writerow([all_sample_idx[i], all_agent_idx[i], ade_flat[i], fde_flat[i], unc_mean_flat[i], unc_final_flat[i]])

    # ============================================================
    # ADE bins
    #
    # Sort by mean trajectory uncertainty
    # Compare with original minADE
    # ============================================================

    ade_sorted_idx = np.argsort(unc_mean_flat)
    ade_bin_indices = np.array_split(ade_sorted_idx, num_bins)

    with open(ade_bin_csv_path, 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(['bin', 'num_agents', 'uncertainty_mean', 'ADE_mean'])

        for bin_id, idx in enumerate(ade_bin_indices):
            csv_writer.writerow([
                bin_id + 1,
                len(idx),
                np.mean(unc_mean_flat[idx]),
                np.mean(ade_flat[idx])
            ])

    # ============================================================
    # FDE bins
    #
    # Sort by final timestep uncertainty
    # Compare with original minFDE
    # ============================================================

    fde_sorted_idx = np.argsort(unc_final_flat)
    fde_bin_indices = np.array_split(fde_sorted_idx, num_bins)

    with open(fde_bin_csv_path, 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(['bin', 'num_agents', 'uncertainty_final', 'FDE_mean'])

        for bin_id, idx in enumerate(fde_bin_indices):
            csv_writer.writerow([
                bin_id + 1,
                len(idx),
                np.mean(unc_final_flat[idx]),
                np.mean(fde_flat[idx])
            ])

    # ============================================================
    # Save summary
    # ============================================================

    with open(summary_csv_path, 'w', newline='') as f:
        csv_writer = writer(f)

        csv_writer.writerow([
            'epoch',
            'mc_samples',
            'num_agents',
            'original_ADE_mean',
            'original_FDE_mean',
            'epistemic_unc_mean',
            'epistemic_unc_final_mean',
            'spearman_uncMean_ADE',
            'spearman_uncFinal_FDE'
        ])

        csv_writer.writerow([
            epoch,
            mc_samples,
            len(ade_flat),
            np.mean(ade_flat),
            np.mean(fde_flat),
            np.mean(unc_mean_flat),
            np.mean(unc_final_flat),
            spearman_ade,
            spearman_fde
        ])

    # ============================================================
    # Save timestep-level uncertainty
    # ============================================================

    np.savez_compressed(
        uncertainty_path,
        epistemic_unc=all_epistemic_unc,
        epistemic_unc_mean=all_unc_mean,
        epistemic_unc_final=all_unc_final,
        original_minADE=all_ade,
        original_minFDE=all_fde
    )

    # ============================================================
    # Print overall results
    # ============================================================

    print('\n==========================================')
    print('Results')
    print('==========================================')

    print('Original minADE: {:.6f}'.format(np.mean(ade_flat)))
    print('Original minFDE: {:.6f}'.format(np.mean(fde_flat)))
    print('Mean epistemic uncertainty: {:.6f}'.format(np.mean(unc_mean_flat)))
    print('Final epistemic uncertainty: {:.6f}'.format(np.mean(unc_final_flat)))
    print('Spearman(UncertaintyMean, ADE): {:.6f}'.format(spearman_ade))
    print('Spearman(UncertaintyFinal, FDE): {:.6f}'.format(spearman_fde))

    # ============================================================
    # Print ADE bins
    # ============================================================

    print('\n==========================================')
    print('ADE Uncertainty Bins')
    print('Mean trajectory uncertainty -> ADE')
    print('==========================================')

    for bin_id, idx in enumerate(ade_bin_indices):
        print('Bin {}/{} | N {:5d} | UncMean {:.6f} | ADE {:.6f}'.format(
            bin_id + 1,
            num_bins,
            len(idx),
            np.mean(unc_mean_flat[idx]),
            np.mean(ade_flat[idx])
        ))

    # ============================================================
    # Print FDE bins
    # ============================================================

    print('\n==========================================')
    print('FDE Uncertainty Bins')
    print('Final timestep uncertainty -> FDE')
    print('==========================================')

    for bin_id, idx in enumerate(fde_bin_indices):
        print('Bin {}/{} | N {:5d} | UncFinal {:.6f} | FDE {:.6f}'.format(
            bin_id + 1,
            num_bins,
            len(idx),
            np.mean(unc_final_flat[idx]),
            np.mean(fde_flat[idx])
        ))

    # ============================================================
    # Files
    # ============================================================

    print('\n[INFO] Saved:')
    print('[1]', agent_csv_path)
    print('[2]', ade_bin_csv_path)
    print('[3]', fde_bin_csv_path)
    print('[4]', summary_csv_path)
    print('[5]', uncertainty_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--dataset', type=str, default='nba', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_nba_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--mc_samples', type=int, default=20, help='number of MC Dropout forward passes')
    parser.add_argument('--num_bins', type=int, default=5, help='number of uncertainty bins')

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    opts = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()