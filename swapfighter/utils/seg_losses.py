import torch
import torch.nn.functional as F

def compute_mask_loss(pos_coef, pos_proto, pos_gt_masks,
                      proto_h, proto_w):
    if pos_coef.numel() == 0 or pos_gt_masks.numel() == 0:
        return torch.zeros((), device=pos_coef.device if pos_coef.numel() else
                          'cpu', requires_grad=False)

    N_pos = pos_coef.shape[0]

    pred_masks = torch.bmm(
        pos_coef.unsqueeze(1),
        pos_proto,
    ).squeeze(1)
    pred_masks = pred_masks.reshape(N_pos, proto_h, proto_w)

    loss = F.binary_cross_entropy_with_logits(
        pred_masks.float(),
        pos_gt_masks.float(),
        reduction='mean',
    )

    if not torch.isfinite(loss):
        return torch.zeros((), device=pos_coef.device, requires_grad=True)
    return loss

__all__ = ['compute_mask_loss']