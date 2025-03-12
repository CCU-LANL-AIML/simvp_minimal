# This file was modified by multiple contributors.
# It includes code originally from the OpenSTL project (https://github.com/chengtan9907/OpenSTL).
# The original code is licensed under the Apache License, Version 2.0.

import numpy as np


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


try:
    from skimage.metrics import structural_similarity as cal_ssim
except:
    cal_ssim = None


def calc_mse(pred, true, spatial_norm=False):
    return MSE(pred, true, spatial_norm)


def calc_mae(pred, true, spatial_norm=False):
    return MAE(pred, true, spatial_norm)


def calc_rmse(pred, true, spatial_norm=False):
    return RMSE(pred, true, spatial_norm)


def calc_ssim(pred, true, clip_range=[0, 1]):
    pred = np.array(pred)
    true = np.array(true)

    pred = np.clip(pred, clip_range[0], clip_range[1])

    if len(pred.shape) == 3:
        pred = pred.reshape(1, 1, *pred.shape)
        true = true.reshape(1, 1, *true.shape)

    ssim_values = []

    for b in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            pred_frame = pred[b, t]
            true_frame = true[b, t]

            pred_t = pred_frame.transpose(1, 2, 0)
            true_t = true_frame.transpose(1, 2, 0)

            data_range = true_t.max() - true_t.min()

            multichannel = pred_frame.shape[0] > 1

            if not multichannel:
                pred_t = pred_t.squeeze(2)
                true_t = true_t.squeeze(2)

            ssim_values.append(cal_ssim(
                pred_t,
                true_t,
                data_range=data_range,
                multichannel=multichannel
            ))

    return np.mean(ssim_values)


def calc_psnr(pred, true, spatial_norm=False):
    return PSNR(pred, true, spatial_norm)