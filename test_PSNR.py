import torch
from utils import util
from tqdm import tqdm
import cv2, os
import models.modules.MV_LKAN as MV_LKAN
import numpy as np

import time

def generate_slices(volume, scale):
    indices_to_remove = [i*scale for i in range(200)]
    indices_to_keep = [idx for idx in range(volume.shape[1]) if idx not in indices_to_remove]
    slice_generated = torch.index_select(volume, dim=1, index=torch.tensor(indices_to_keep).to(volume.device))
    return slice_generated


def main():

    scale = 6
    dataset = 'IXI'
    save_result = False
    model_name = 'braints_x'+str(scale)+'_CFFNet_volume'

    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    #### create train and val dataloader
    if dataset == 'BrainTS':
        from data.brainseg_dataset_test import brainseg_test as D
        testset_path = "/data3/lpc/dataset/BrainTS/T2_test_nii"
    elif dataset == 'IXI':
        from data.brainseg_dataset_test import brainseg_test as D
        testset_path = "/data3/lpc/dataset/IXI/nii-splitted/test/T2"

    # 保存图像
    if save_result:
        save_path_1 = '/data3/lpc/program/VolumeSR/result_img/'+ model_name
        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)
    
    val_set = D(scale=scale,test_path=testset_path)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))  
   
    model_path = '....'
    model = MV_LKAN.MV_LKAN(scale=scale).cuda()
    
    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        #### validation
          
        avg_psnr_im1 = 0.0
        avg_ssim_im1 = 0.0
        avg_rmse_im1 = 0.0
        idx = 0
        psnr = []
        ssim = []
        rmse = []

        for i,val_data in enumerate(tqdm(val_loader)): 

            volume_lr = val_data['volume_LQ'].cuda()
            volume_gt = val_data['volume_GT'].cuda()
            volume_ref = val_data['volume_ref'].cuda()

            # if save_result:
            #     cv2.imwrite(os.path.join(save_path_1, '{:02d}_img_000.png'.format(i)), volume_lr[0,0,0].cpu().detach().numpy()*255.)

            slice_len = 24
            for j in range(volume_gt.shape[1]//slice_len):
                seq_inputs = volume_gt[:,slice_len*j:slice_len*(j+1)+1]  ##[1->37][37->37+36][37+36->37+36+36]
                ref_inputs = volume_ref[:,slice_len*j:slice_len*(j+1)+1]
                img_inputs = seq_inputs[:, ::scale]
                # print(img_inputs.shape[1])
                sr_img,_ = model(img_inputs, ref_inputs)
                sr_img = sr_img[0,:,0].cpu().detach().numpy()*255.
                im_gt = generate_slices(seq_inputs,scale)
                im_gt = im_gt[0,:,0].cpu().detach().numpy()*255.
                
                for k in range(sr_img.shape[0]):
                    # calculate PSNR
                    cur_psnr_im1 = util.calculate_psnr(sr_img[k], im_gt[k])
                    psnr.append(cur_psnr_im1)
                    avg_psnr_im1 += cur_psnr_im1
                    cur_ssim_im1 = util.calculate_ssim(sr_img[k], im_gt[k])
                    ssim.append(cur_ssim_im1)
                    avg_ssim_im1 += cur_ssim_im1
                    cur_rmse_im1 = util.calculate_rmse(sr_img[k], im_gt[k])
                    rmse.append(cur_rmse_im1)
                    avg_rmse_im1 += cur_rmse_im1
                    print('########', i,j,k,cur_psnr_im1,cur_ssim_im1)
                    idx += 1

                    # if save_result:
                    #     cv2.imwrite(os.path.join(save_path_1, '{:02d}_img_{:03d}.png'.format(i,(j*scale+1+k))), sr_img[k])
                
                # if save_result:
                #     cv2.imwrite(os.path.join(save_path_1, '{:02d}_img_{:03d}.png'.format(i,(j*scale+scale))), img_inputs[0,-1,0].cpu().detach().numpy()*255.)

                    
        
        avg_psnr_im1 = avg_psnr_im1 / idx
        avg_ssim_im1 = avg_ssim_im1 / idx
        avg_rmse_im1 = avg_rmse_im1 / idx

        # log
        std_psnr = np.sqrt(np.mean((psnr - np.mean(psnr)) ** 2))
        std_ssim = np.sqrt(np.mean((ssim - np.mean(ssim)) ** 2))
        std_rmse = np.sqrt(np.mean((rmse - np.mean(rmse)) ** 2))
        print("# image1 Validation # PSNR: {:.6f}, std: {:.5f}".format(avg_psnr_im1, std_psnr))
        print("# image1 Validation # SSIM: {:.6f}, std: {:.5f}".format(avg_ssim_im1, std_ssim))
        print("# image1 Validation # RMSE: {:.6f}, std: {:.5f}".format(avg_rmse_im1, std_rmse))


### CUDA_VISIBLE_DEVICES=2 python test_PSNR.py
if __name__ == '__main__':
    main()

