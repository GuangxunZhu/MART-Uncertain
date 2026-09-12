import os
import random
import argparse

from csv import writer

import torch
import numpy as np

from torch import optim
from torch.optim import lr_scheduler

from utils import *
from models.mart import CVAEMart
from loaders.dataloader_eth import TrajectoryDataset


def log_print(msg, log_file='log.txt'):
    print(msg)

    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')

def calculate_loss_pred(pred,target,batch_size):
    loss = (target-pred).pow(2).sum()
    loss /= batch_size
    loss /= pred.shape[1]
    return loss
def calculate_loss_kl(qz_distribution,pz_distribution,batch_size,agent_num,min_clip):
    loss = qz_distribution.kl(pz_distribution).sum()
    loss /= (batch_size * agent_num)
    loss_clamp = loss.clamp_min_(min_clip)
    return loss_clamp

def calculate_loss_recover(pred,target,batch_size):
    loss = (target-pred).pow(2).sum()
    loss /= batch_size
    loss /= pred.shape[1]
    return loss

def calculate_loss_diverse(pred,target,batch_size):
    diff = target.unsqueeze(1) - pred
    avg_dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss = avg_dist.min(dim=1)[0]
    loss = loss.mean() 
    return loss



def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)
    log_print(f'[INFO] The seed is: {seed}')

    # print('[INFO] The seed is:', seed)

    data_root = os.path.join('./datasets/ethucy', args.dataset)

    if not args.test:
        dataset_train = TrajectoryDataset(args, os.path.join(data_root, 'train'), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
        
    dataset_test = TrajectoryDataset(args, os.path.join(data_root, 'test'), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    # model = MART(opts).cuda()
    model = CVAEMart(opts).cuda()
    log_print(f'[INFO] Model:\n{model}')
    log_print(f'[INFO] Model params: {sum(p.numel() for p in model.parameters())}')
    # print('[INFO] Model params: {}'.format(sum(p.numel() for p in model.parameters())))

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
        # print('[INFO] Loading model from:', model_path)
        log_print(f'[INFO] Loading model from: {model_path}')
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()

    results = {'epochs': [], 'losses': []}
    best_val_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    log_print(f'[INFO] The seed is: {seed}')
    
    for epoch in range(0, opts.num_epochs):
        train(epoch, model, optimizer, loader_train)
        test_loss, ade = test(epoch, model, loader_test)
        results['epochs'].append(epoch)
        results['losses'].append(test_loss)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_ade = ade
            best_epoch = epoch
            file_path = os.path.join(model_save_dir, str(args.dataset) + '_ckpt_best.pth')
            torch.save(state, file_path)
        print('[INFO] Best {} Loss: {:.5f} \t Best ade: {:.5f} \t Best epoch {}\n'.format('TEST', best_val_loss, best_ade, best_epoch))
        log_print(f'[INFO] Best TEST Loss: {best_val_loss:.5f} \t Best ade: {best_ade:.5f} \t Best epoch {best_epoch}\n')
       

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
    batch_count, divider = 0, 0
    is_first_loss = True

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        batch_count += 1
        divider += 1
        
        x_abs, y = data
        x_abs, y = x_abs.cuda(), y.cuda()        
        
        batch_size, num_agents, length, _ = x_abs.size()

        x_rel = torch.zeros_like(x_abs)
        x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
        x_rel[:, :, 0] = x_rel[:, :, 1]

        y_rel = torch.zeros_like(y)
        y_rel[:, :, 1:] = y[:, :, 1:] - y[:, :, :-1]
        y_rel[:, :, 0] = y[:, :, 0] - x_abs[:, :, -1]
        
        # y_pred = model(x_abs, x_rel)
        pred_endpoint, diverse_pred_endpoint, pred_traj_final, recover_traj_final, diverse_pred_traj_final, pz_distribution, qz_distribution, pz_distribution_social, qz_distribution_social = model(x_abs, x_rel, y, y_rel)
        # y_pred = diverse_pred_traj_final.view(batch_size, num_agents, -1, opts.future_length, 2)
        # if opts.pred_rel:
        #     cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
        #     y_pred = torch.cumsum(y_pred, dim=3) + cur_pos
        
        loss_pred = calculate_loss_pred(pred_endpoint, y.view(batch_size * num_agents, -1, 2)[:, [-1]], batch_size)
        loss_kl = calculate_loss_kl(qz_distribution, pz_distribution, batch_size, num_agents, min_clip=2.0)
        loss_diverse = calculate_loss_diverse(diverse_pred_endpoint, y[:, :, -1:, :].view(batch_size* num_agents, 1, 1, 2), batch_size)

        loss_pred_final = calculate_loss_pred(pred_traj_final, y.view(batch_size * num_agents, -1, 2), batch_size)
        loss_recover_final = calculate_loss_recover(recover_traj_final, x_abs.view(batch_size * num_agents, opts.past_length, -1), batch_size)
        loss_kl_final = calculate_loss_kl(qz_distribution_social, pz_distribution_social, batch_size, num_agents, min_clip=2.0)
        loss_diverse_final = calculate_loss_diverse(diverse_pred_traj_final, y.view(batch_size * num_agents, -1, 2), batch_size)

        # total_loss = loss_pred +loss_kl +  loss_diverse + loss_pred_final + loss_recover_final + loss_kl_final + loss_diverse_final + loss_coarse_future_align
        base_loss = (
            loss_pred
            + loss_kl
            + loss_diverse
            + loss_pred_final
            + loss_recover_final
            + loss_kl_final
            + loss_diverse_final
        )

        # lambda_align_max = 1e-4
        # warmup_start = 20
        # warmup_end = 50

        # if epoch < warmup_start:
        #     lambda_align = 0.0
        # else:
        #     lambda_align = lambda_align_max * min(
        #         1.0,
        #         (epoch - warmup_start) / (warmup_end - warmup_start)
        #     )

        # total_loss = base_loss + lambda_align * loss_coarse_future_align
        total_loss = base_loss
        # y = y[:, :, None, :, :]
        
        # total_loss = torch.mean(torch.min(torch.mean(torch.norm(y_pred - y, dim=-1), dim=3), dim=2)[0]) # for all agents
        avg_meter['loss'] += total_loss.item() * batch_size * num_agents
        avg_meter['counter'] += (batch_size * num_agents)

        if is_first_loss:
            loss = total_loss
            is_first_loss = False
        else:
            loss += total_loss

        if batch_count % opts.batch_size == 0: # or i == loader_len - 1:
            loss = loss / divider
            is_first_loss = True
            
            loss.backward()
            if opts.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_grad)
                
            optimizer.step()

            th = get_th(opts, model.past_traj_encoder)
            log_print('[{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Pred_Loss: {:03f} | loss_endpoint_kl: {:03f}  | loss_endpoint_pred: {:03f} | loss_endpoint_diverse: {:03f} | loss_recover: {:03f} | loss_kl: {:03f} | loss_diverse: {:03f} |LR: {}'
                  .format(args.dataset.upper(), epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), loss_pred_final.item(),  loss_kl.item(), loss_pred.item(), loss_diverse.item(), loss_recover_final.item(), loss_kl_final.item(), loss_diverse_final.item(), optimizer.param_groups[0]['lr']))
            # print('[{}][{}] Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Loss: {:03f} | Threshold: {} | LR: {}'
            #       .format(args.dataset.upper(), 'TRAIN', epoch + 1, opts.num_epochs, i + 1, loader_len, total_loss.item(), th, optimizer.param_groups[0]['lr']))
    return avg_meter['loss'] / avg_meter['counter']


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    
    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1]

            y_rel = torch.zeros_like(y)
            y_rel[:, :, 1:] = y[:, :, 1:] - y[:, :, :-1]
            y_rel[:, :, 0] = y[:, :, 0] - x_abs[:, :, -1]
            
            # y_pred = model(x_abs, x_rel)
            pred_endpoint, diverse_pred_endpoint, pred_traj_final, recover_traj_final, diverse_pred_traj_final, pz_distribution, qz_distribution, pz_distribution_social, qz_distribution_social = model(x_abs, x_rel, y, y_rel)

            y_pred = diverse_pred_traj_final.view(batch_size, num_agents, -1, opts.future_length, 2)

            # if opts.pred_rel:
            #     cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
            #     y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = np.array(y_pred.cpu()) # B, N, 20, T, 2
            y = np.array(y.cpu()) # B, N, T, 2
            y = y[:, :, None, :, :]
            
            ade = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += (num_agents * batch_size)
    
    th = get_th(opts, model.past_traj_encoder)
    log_print('\n[{}][{}] Epoch {} th: {}'.format(args.dataset.upper(), 'TEST', epoch, th))
    log_print('[{}][{}] minADE/minFDE: {:.2f}/{:.2f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MART for Trajectory Prediction')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='eth', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='configs/mart_eth_reproduce.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    opts = load_config(args.config)
    if args.dataset == 'eth' or args.dataset == 'univ':
        opts.lr = 0.001
    elif args.dataset == 'zara1' or args.dataset == 'zara2':
        opts.lr = 0.0012
    else:
        opts.lr = 0.0018
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()