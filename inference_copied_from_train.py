import os
import sys
import math

from scipy.stats import spearmanr, pearsonr
import numpy as np
import wandb
from einops import rearrange

import torch
from torch.utils.data import DataLoader
import torchvision

from network.network_utils import build_model
from network.optimizer_utils import get_optimizer, get_scheduler
from dataloaders import dataloader_gen

from utils.loss_utils import compute_center_loss_v2, compute_rpt_direction_v1, compute_mae_loss_v3, compute_metric_loss_v1
from utils.train_utils import get_batches_v7_mask_single, Maxlloyd, get_left_right_idxs, get_pairs_equally_fast_v2

from utils.test_utils import compute_score_v1

from utils.util import load_model, save_model
from utils.util import set_wandb, tensor2np, write_log


def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    train_dataset, test_ref_loader, test_loader = dataloader_gen.gen_dataloader(cfg)

    train_dataset_moses = train_dataset.df['MOS'].values
    cfg.n_scores = len(np.unique(train_dataset_moses))

    # pivot score setting
    maxlloyd = Maxlloyd(train_dataset_moses, rpt_num=cfg.spv_num)
    cfg.score_pivot_score = maxlloyd.get_new_rpt_scores()
    cfg.reference_point_num = len(cfg.score_pivot_score)

    cfg.log_file = cfg.log_configs()

    write_log(cfg.log_file, f'[*] {cfg.n_scores} scores exist.')
    write_log(cfg.log_file, f'[*] {cfg.reference_point_num} reference points.')

    net = build_model(cfg)

    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(net)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        net = net.cuda()

    optimizer = get_optimizer(cfg, net)
    lr_scheduler = get_scheduler(cfg, optimizer)

    if cfg.load:
        load_model(cfg, net, optimizer=optimizer, load_optim_params=False)

    if cfg.test_first:
        net.eval()
        srcc, pcc, mae = evaluation(cfg, net, test_ref_loader, test_loader)
        sys.stdout.write(f'[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

    best_srcc = 0.85
    best_pcc = 0.85

    best_srcc_total_results = []
    best_pcc_total_results = []

    best_srcc_epoch = -1
    best_pcc_epoch = -1

    log_dict = dict()
    for epoch in range(0, cfg.epoch):
        net.train()
        net.encoder.eval()

        if (epoch + 1) <= 5:
            uniform_select = True
        else:
            uniform_select = False

        train_dataset.get_pair_lists(batch_size=cfg.batch_size, batch_list_len=cfg.im_list_len, im_num=cfg.im_num, uniform_select=uniform_select)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)

        mae_loss, center_loss, order_loss, metric_loss = train(cfg, net, optimizer, train_loader, epoch)
        write_log(cfg.log_file, f'\nEpoch: {(epoch + 1):d} MAE Loss: {mae_loss:.3f}, Center Loss: {center_loss:.3f}, Order Loss: {order_loss:.3f}, Metric Loss: {metric_loss:.3f}\n')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['LR'] = lr_scheduler.get_lr()[0] if lr_scheduler else cfg.lr

        if ((epoch + 1) == 1) | (((epoch + 1) >= cfg.start_eval) & ((epoch + 1) % cfg.eval_freq == 0)):

            net.eval()
            srcc, pcc, mae = evaluation(cfg, net, test_ref_loader, test_loader)
            sys.stdout.write(f'\n[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

            if srcc > best_srcc:
                best_srcc = srcc
                best_srcc_epoch = epoch
                best_srcc_total_results = [srcc, pcc, mae]
                save_model(cfg, net, optimizer, epoch, [srcc, pcc, mae], criterion='SRCC')

            if pcc > best_pcc:
                best_pcc = pcc
                best_pcc_epoch = epoch
                best_pcc_total_results = [srcc, pcc, mae]
                save_model(cfg, net, optimizer, epoch, [srcc, pcc, mae], criterion='PCC')

            if cfg.wandb:
                log_dict['Test/SRCC'] = srcc
                log_dict['Test/PCC'] = pcc
                log_dict['Test/MAE'] = mae

        if cfg.wandb:
            wandb.log(log_dict)

        print('lr: %.6f' % (optimizer.param_groups[0]['lr']))
        if lr_scheduler:
            lr_scheduler.step()

    write_log(cfg.log_file, 'Training End')
    write_log(cfg.log_file,
              'Best SRCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_srcc_epoch, best_srcc_total_results[0], best_srcc_total_results[1], best_srcc_total_results[2]))
    write_log(cfg.log_file,
              'Best PCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_pcc_epoch, best_pcc_total_results[0], best_pcc_total_results[1], best_pcc_total_results[2]))
    print(cfg.save_folder)



def evaluation(cfg, net, ref_data_loader, data_loader):
    net.eval()
    test_mos_gt = data_loader.dataset.df_test['MOS'].values

    preds_list = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            # Extract features of auxiliary images
            f_list = []
            for idx, sample in enumerate(ref_data_loader):
                if idx % 1 == 0:
                    sys.stdout.write(f'\rExtract Aux Img Features... [{idx + 1}/{len(ref_data_loader)}]')

                image = sample[f'img'].cuda()
                f = net('extraction', {'img': image})
                f_list.append(f)

            aux_f = torch.cat(f_list)
            aux_f = aux_f.squeeze()
            aux_f = aux_f.transpose(1, 0)

            # Extract features of test images
            test_f_list = []
            for idx, sample in enumerate(data_loader):

                if idx % 1 == 0:
                    sys.stdout.write(f'\rExtract Test Img Features... [{idx + 1}/{len(data_loader)}]')

                image = sample[f'img'].cuda()
                f = net('extraction', {'img': image})
                f_hflip = net('extraction', {'img': torchvision.transforms.functional.hflip(image)})

                test_f_list.append(f)
                test_f_list.append(f_hflip)

            test_f = torch.cat(test_f_list)
            test_f = test_f.squeeze()
            test_f = rearrange(test_f, '(N Cr) C -> N Cr C', N=len(test_mos_gt), C=cfg.reduced_dim).mean(1)
            test_f = test_f.transpose(1, 0)

            # Set # of iterations
            n_iter = int(math.ceil(len(test_mos_gt) / cfg.test_batch_size))
            crop_num = 1
            start = 0

            for idx in range(n_iter):
                if idx % 1 == 0:
                    sys.stdout.write(f'\rTesting... [{idx + 1}/{n_iter}]')

                batch = min(cfg.test_batch_size, len(test_mos_gt) - len(preds_list))

                f = torch.cat([aux_f.unsqueeze(0).repeat(batch, 1, 1),
                               rearrange(test_f[:, start:start + (batch * crop_num)], 'C (N L) -> N C L', N=batch, L=crop_num)], dim=-1)

                # Obtain updated features and score pivots
                f, score_pivots = net('get_cluster', {'f': f})

                # Estimate quality scores
                preds = compute_score_v1(embs=rearrange(f, 'b c l -> b l c')[:, -1:],
                                         spv=rearrange(score_pivots, 'b c l -> b l c'),
                                         emb_scores=torch.tensor(test_mos_gt[start:(start + batch)].reshape(-1, 1)).cuda().float(),
                                         spv_scores=cfg.score_pivot_score,
                                         )

                preds_list.extend(preds.tolist())
                start += (batch * crop_num)

    preds_np = np.array(preds_list)

    srcc = spearmanr(preds_np, test_mos_gt)[0]
    pcc = pearsonr(preds_np, test_mos_gt)[0]
    mae = np.abs(preds_np - test_mos_gt).mean()

    write_log(cfg.log_file, f'\nTest MAE: {mae: .4f} SRCC: {srcc: .4f} PCC: {pcc: .4f}')

    return srcc, pcc, mae



if __name__ == "__main__":
    from configs.config_v1 import ConfigV1 as Config

    cfg = Config()


    # Set inference-specific parameters
    cfg.load = True  # Enable model loading
    # cfg.reference_point_num = 101
    cfg.dataset_name = 'SPAQ'  # or 'KonIQ10K'
    cfg.ckpt_file = 'SRCC_Epoch_94_SRCC_0.9246_PCC_0.9265_MAE_6.1870.pth'
    cfg.init_model = f'./ckpt/{cfg.dataset_name}/Split_{cfg.split}/{cfg.ckpt_file}'
    cfg.dataset_root = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test'
    cfg.datasplit_root = r"C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\pictime"

    # Run inference
    main(cfg)