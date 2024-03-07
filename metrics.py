import math
import numpy as np
from scipy import stats

# Affine Invariant 
# TODO: Hparam tune?
def depth_from_disp(disp_map):
    # -20 to +20
    focus = 50 
    baseline = 6.25 
    # infocus = 400 # is infinity
    pixel_pitch = 0.334 # 0.875*10.36 * 1e-3 (mm)  
    # disp = baseline * focus * (1/depth - 1/infcous) 

    depth = 1/ (((pixel_pitch*disp_map)/(baseline*focus)))
    depth_scaled = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth,depth_scaled


def compute_AI_errors(gt, pred):
    ai_mae = np.mean(np.abs(gt - pred))
    ai_rmse = np.sqrt(np.mean((gt - pred) ** 2))
    # 1 - spearman's correlation coefficient
    ai_spearman = 1 - stats.spearmanr(gt, pred)
    return {"metrics/ai_mae": ai_mae, "metrics/ai_rmse": ai_rmse, "metrics/ai_spearman": ai_spearman}


def compute_report_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    eps = 1e-7 # for circumventing non-zero division
    abs_rel = np.mean(np.abs(gt - pred) / (gt + eps))

    sq_rel = np.mean(((gt - pred)**2) / (gt + eps))

    return {"metrics/abs_rel": abs_rel, "metrics/sq_rel": sq_rel, "metrics/rmse": rmse, 
            "metrics/log_rmse": rmse_log, "metrics/del_1": a1, "metrics/del_2": a2, "metrics/del_3": a3}
