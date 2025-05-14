import os
import sys
import torch
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from einops import rearrange
import math
from scipy.stats import spearmanr, pearsonr

# Import QCN modules
from configs.config_v1 import ConfigV1 as Config
from utils.test_utils import compute_score_v1
from utils.util import load_model, write_log
from dataloaders.dataloader_gen import gen_dataloader_for_inference
from utils.train_utils import Maxlloyd

def inference(cfg):
    # Set up logging
    os.makedirs(cfg.save_folder, exist_ok=True)
    cfg.log_file = open(os.path.join(cfg.save_folder,f'inference_log_{cfg.dataset_name}_{cfg.video_segment}.txt'), 'w')

    # Generate dataloaders
    test_ref_loader, test_loader = gen_dataloader_for_inference(cfg)

    # Build model
    from network.network_utils import build_model

    net = build_model(cfg)

    # Load pretrained weights
    if cfg.load:
        load_model(cfg, net, optimizer=None, load_optim_params=False)
    else:
        print("Error: No model checkpoint specified. Set cfg.load=True and cfg.init_model path.")
        return

        # Run inference
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    results = evaluate_images(cfg, net, test_ref_loader, test_loader)

    # Save results to CSV
    save_results_to_csv(cfg, results)

    print(f"Inference completed. Results saved to {cfg.save_folder}/image_scores.csv")

    return results


def evaluate_images(cfg, net, ref_data_loader, data_loader):
    net.eval()
    test_mos_gt = data_loader.dataset.df_test['MOS'].values
    image_names = data_loader.dataset.df_test['image_name'].values

    preds_list = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Extract features of auxiliary images
            print("Extracting auxiliary image features...")
            f_list = []
            for idx, sample in enumerate(ref_data_loader):
                if idx % 10 == 0:
                    sys.stdout.write(f'\rExtract Aux Img Features... [{idx + 1}/{len(ref_data_loader)}]')

                image = sample[f'img'].cuda()
                f = net('extraction', {'img': image})
                f_list.append(f)

            aux_f = torch.cat(f_list)
            aux_f = aux_f.squeeze()
            # aux_f = aux_f.unsqueeze(0)
            aux_f = aux_f.transpose(1, 0)
            print("\nAuxiliary features extracted.")

            # Extract features of test images
            print("Extracting test image features...")
            test_f_list = []
            for idx, sample in enumerate(data_loader):
                if idx % 10 == 0:
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
            print("\nTest features extracted.")

            # Set # of iterations
            n_iter = int(math.ceil(len(test_mos_gt) / cfg.test_batch_size))
            crop_num = 1
            start = 0

            print("Computing quality scores...")
            for idx in range(n_iter):
                if idx % 1 == 0:
                    sys.stdout.write(f'\rTesting... [{idx + 1}/{n_iter}]')

                batch = min(cfg.test_batch_size, len(test_mos_gt) - len(preds_list))

                f = torch.cat([aux_f.unsqueeze(0).repeat(batch, 1, 1),
                               rearrange(test_f[:, start:start + (batch * crop_num)], 'C (N L) -> N C L', N=batch,
                                         L=crop_num)], dim=-1)

                # Obtain updated features and score pivots
                f, score_pivots = net('get_cluster', {'f': f})

                # Estimate quality scores
                preds = compute_score_v1(embs=rearrange(f, 'b c l -> b l c')[:, -1:],
                                         spv=rearrange(score_pivots, 'b c l -> b l c'),
                                         emb_scores=torch.tensor(
                                             test_mos_gt[start:(start + batch)].reshape(-1, 1)).cuda().float(),
                                         spv_scores=cfg.score_pivot_score,
                                         )

                preds_list.extend(preds.tolist())
                start += (batch * crop_num)

    preds_np = np.array(preds_list)

    # Calculate metrics if ground truth is available
    if not np.all(test_mos_gt == 0):  # Check if we have real ground truth
        srcc = spearmanr(preds_np, test_mos_gt)[0]
        pcc = pearsonr(preds_np, test_mos_gt)[0]
        mae = np.abs(preds_np - test_mos_gt).mean()

        print(f"\nTest Metrics - SRCC: {srcc:.4f}, PCC: {pcc:.4f}, MAE: {mae:.4f}")
        write_log(cfg.log_file, f'\nTest Metrics - SRCC: {srcc:.4f}, PCC: {pcc:.4f}, MAE: {mae:.4f}')
    else:
        print("\nNo ground truth available for metric calculation.")

        # Prepare results
    results = {
        'image_name': image_names,
        'predicted_score': preds_np,
        'ground_truth': test_mos_gt
    }

    return results


def save_results_to_csv(cfg, results):
    import shutil
    # Create DataFrame with results
    df = pd.DataFrame({
        'image_name': results['image_name'],
        'predicted_score': results['predicted_score'],
        'ground_truth': results['ground_truth']
    })

    # Sort by predicted score (descending)
    df_sorted = df.sort_values(by='predicted_score', ascending=False)

    # Save to CSV
    csv_path = os.path.join(cfg.save_folder, f'image_scores_{cfg.dataset_name}_{cfg.video_segment}.csv')
    df_sorted.to_csv(csv_path, index=False)

    # Log top and bottom 5 images
    top_5_path = os.path.join(cfg.save_top_bottom_5_results, 'top_5')
    bottom_5_path = os.path.join(cfg.save_top_bottom_5_results, 'bottom_5')
    os.makedirs(top_5_path, exist_ok=True)
    os.makedirs(bottom_5_path, exist_ok=True)

    write_log(cfg.log_file, "\nTop 5 images by quality score:")
    for _, row in df_sorted.head(5).iterrows():
        write_log(cfg.log_file, f"Image: {row['image_name']}, Score: {row['predicted_score']:.4f}")
        shutil.copy(os.path.join(cfg.images_folder_original_path, row['image_name']), os.path.join(top_5_path, row['image_name']))

    write_log(cfg.log_file, "\nBottom 5 images by quality score:")
    for _, row in df_sorted.tail(5).iterrows():
        write_log(cfg.log_file, f"Image: {row['image_name']}, Score: {row['predicted_score']:.4f}")
        shutil.copy(os.path.join(cfg.images_folder_original_path, row['image_name']), os.path.join(bottom_5_path, row['image_name']))



if __name__ == "__main__":
    # Load configuration
    cfg = Config()

    # Set inference-specific parameters
    cfg.load = True  # Enable model loading
    cfg.dataset_name = 'KonIQ10K'  # 'SPAQ'  or 'KonIQ10K'
    cfg.ckpt_file = 'SRCC_Epoch_89_SRCC_0.9349_PCC_0.9446_MAE_3.7669.pth'
    cfg.init_model = f'./ckpt/{cfg.dataset_name}/Split_{cfg.split}/{cfg.ckpt_file}'
    cfg.dataset_root = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test'
    cfg.datasplit_root = r"C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\pictime"
    cfg.video_segment = 'segment_010'

    # pivot score setting
    # train_dataset_moses = pd.read_excel(fr"C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\{cfg.dataset_name}\{cfg.dataset_name}_train_split_1.xlsx")['MOS'].values
    train_dataset_moses = pd.read_csv(fr"C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\{cfg.dataset_name}\{cfg.dataset_name}_train_split_1.csv")['MOS'].values
    maxlloyd = Maxlloyd(train_dataset_moses, rpt_num=cfg.spv_num)
    cfg.score_pivot_score = maxlloyd.get_new_rpt_scores()
    cfg.reference_point_num = len(cfg.score_pivot_score)

    from my_utils.resize_images_preprocess import resize_images
    from my_utils.prepare_csv_for_inference import create_mos_csv
    cfg.video_name = 'Film'
    cfg.save_folder = os.path.join(cfg.save_folder, cfg.video_name)
    segments_names = os.listdir(fr'C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{cfg.video_name}\frames\frames of segments')
    for it, seg_num in enumerate(segments_names):
        if it <10:
            continue
        cfg.video_segment = seg_num

        # saveing top 5 and bottom 5 results
        cfg.save_top_bottom_5_results = fr'C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{cfg.video_name}\frames\QCN\{cfg.dataset_name}\{cfg.video_segment}'
        cfg.images_folder_original_path = fr'C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{cfg.video_name}\frames\frames of segments\{cfg.video_segment}'

        # resize the images
        resize_images(fr"C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{cfg.video_name}\frames\frames of segments\{seg_num}",
                      fr"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\{cfg.dataset_name}\{cfg.video_name}\{seg_num}_test",
                      512, 384)

        # prepare csv for frames
        image_folder = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\{cfg.dataset_name}\{cfg.video_name}\{seg_num}_test'
        output_folder = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\pictime\{cfg.video_name}'
        output_filename = f"{seg_num}_test.csv"
        create_mos_csv(image_folder, output_folder, output_filename)

        # Run inference
        results = inference(cfg)
        print(fr'Done {seg_num}')

    print("Done")