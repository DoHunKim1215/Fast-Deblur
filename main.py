import argparse
import os

import numpy
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from torch.backends import cudnn
from torchvision.transforms import functional as F

from models.colorization_model import create_colorization_model
from models.deblur_model import create_deblur_model
from preprocess.data_load import test_dataloader
from utils.adder import Adder
from utils.color_to_hint import get_color_hint_evenly_faster
from utils.converter import rgb2lab, lab2rgb


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Test
    parser.add_argument('--colorization_model', type=str,
                        default='models\\colorization_best.pkl')
    parser.add_argument('--deblur_model', type=str,
                        default='models\\deblur_best.pkl')
    parser.add_argument('--data_dir', type=str, default='dataset\\GoPro\\test')
    parser.add_argument('--result_dir', type=str, default='results\\images')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()

    print(args)

    return args


if __name__ == '__main__':
    args = get_args()

    # Turn on benchmark mode to perform effective calculation
    cudnn.benchmark = True

    # Load Colorization Model
    color_model = create_colorization_model()

    if torch.cuda.is_available():
        color_model.cuda()

    state_dict = torch.load(args.colorization_model)
    color_model.load_state_dict(state_dict['model'])
    color_model.eval()

    # Load Deblur Model
    deblur_model = create_deblur_model()

    if torch.cuda.is_available():
        deblur_model.cuda()

    state_dict = torch.load(args.deblur_model)
    deblur_model.load_state_dict(state_dict['model'])
    deblur_model.eval()

    # Get dataloader
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)

    # CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # elapsed time adder
    adder1 = Adder()
    adder2 = Adder()
    adder3 = Adder()

    # inference time checker
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()
        mem_adder = Adder()

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            blur_img, sharp_img, name = data
            blur_img = blur_img.to(device)
            sharp_img = sharp_img.to(device)

            blur_img = blur_img[:, :, :256, :256]
            sharp_img = sharp_img[:, :, :256, :256]

            mem_usage = torch.cuda.memory_allocated() / 1024 / 1024
            mem_adder(mem_usage)

            blur_lab = rgb2lab(blur_img)

            starter.record()

            gray_sharp = deblur_model(blur_lab)

            ender.record()
            torch.cuda.synchronize()
            elapsed1 = starter.elapsed_time(ender)

            starter.record()

            hint = get_color_hint_evenly_faster(blur_lab, patch_size=16)

            ender.record()
            torch.cuda.synchronize()
            elapsed2 = starter.elapsed_time(ender)

            starter.record()

            pred_ab = color_model(gray_sharp, hint)
            pred_img = lab2rgb(torch.cat([gray_sharp, pred_ab], dim=1))

            ender.record()
            torch.cuda.synchronize()
            elapsed3 = starter.elapsed_time(ender)

            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender)

            adder1(elapsed1)
            adder2(elapsed2)
            adder3(elapsed3)

            pred_clip = torch.clamp(pred_img, 0, 1)

            pred_numpy: numpy.ndarray = pred_clip.squeeze(0).cpu().numpy()
            label_numpy: numpy.ndarray = sharp_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)

            pred_numpy = pred_numpy.transpose((1, 2, 0))
            label_numpy = label_numpy.transpose((1, 2, 0))

            ssim_, _ = ssim(pred_numpy, label_numpy, channel_axis=-1, full=True, data_range=1.0)
            psnr_adder(psnr)
            ssim_adder(ssim_)
            print('%d iter PSNR: %.2f SSIM: %.8f time: %f mem: %f' % (
                iter_idx + 1, psnr, ssim_, elapsed1 + elapsed2 + elapsed3, mem_usage))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.8f' % (ssim_adder.average()))
        print('The average mem is %f' % (mem_adder.average()))
        print("Average time1: %f" % adder1.average())
        print("Average time2: %f" % adder2.average())
        print("Average time3: %f" % adder3.average())
        print("Average time: %f" % (adder1.average() + adder2.average() + adder3.average()))
