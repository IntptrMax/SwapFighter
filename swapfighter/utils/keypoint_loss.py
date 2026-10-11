import torch
import torch.nn.functional as F

def compute_keypoint_loss(kpt_list, targets,
                          pos_batch, pos_point, pos_gt,
                          strides, img_size, num_keypoints,
                          coord_weight=1.0):
    B = kpt_list[0].shape[0]
    device = kpt_list[0].device

    all_kpt = []
    for kpt in kpt_list:
        b, c, h, w = kpt.shape
        all_kpt.append(kpt.reshape(b, c, -1))
    all_kpt = torch.cat(all_kpt, dim=2)

    if pos_batch is None or pos_batch.numel() == 0:
        return all_kpt.sum() * 0.0

    N_pos = pos_batch.numel()

    pos_pred = all_kpt[pos_batch, :, pos_point]
    pos_pred = pos_pred.reshape(N_pos, num_keypoints, 2)

    gt_list = []
    for i in range(N_pos):
        b = int(pos_batch[i])
        g = int(pos_gt[i])
        gt_list.append(targets[b]['keypoints'][g])
    gt_kpt = torch.stack(gt_list, dim=0).to(device)

    gt_norm = gt_kpt[..., :2] / img_size
    gt_v = gt_kpt[..., 2]

    vis_mask = (gt_v > 0).float()

    l1 = F.l1_loss(pos_pred, gt_norm, reduction='none').mean(dim=-1)
    loss = (l1 * vis_mask).sum() / (vis_mask.sum() + 1e-6)

    if not torch.isfinite(loss):
        return pos_pred.sum() * 0.0

    return loss * coord_weight

__all__ = ['compute_keypoint_loss']