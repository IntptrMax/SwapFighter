import numpy as np
import cv2
import torch
import torch.nn.functional as F

def polygons_to_mask(polygons, img_h, img_w, downsample=4):
    h_small = img_h // downsample
    w_small = img_w // downsample
    mask = np.zeros((h_small, w_small), dtype=np.float32)

    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        pts = (poly / downsample).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)

    return mask

def polygons_to_masks_per_gt(polygons, img_h, img_w, downsample=4):
    h_small = img_h // downsample
    w_small = img_w // downsample
    masks = np.zeros((len(polygons), h_small, w_small), dtype=np.float32)

    for i, poly in enumerate(polygons):
        if poly is None or len(poly) < 3:
            continue
        pts = (poly / downsample).astype(np.int32)
        cv2.fillPoly(masks[i], [pts], 1.0)

    return masks

def crop_mask_to_box_torch(masks, boxes):
    if masks.numel() == 0 or boxes.numel() == 0:
        return masks

    N, H, W = masks.shape
    device = masks.device

    x1 = boxes[:, 0].clamp(0, W).long()
    y1 = boxes[:, 1].clamp(0, H).long()
    x2 = boxes[:, 2].clamp(0, W).long()
    y2 = boxes[:, 3].clamp(0, H).long()

    rows = torch.arange(H, device=device).view(1, H, 1).float()
    cols = torch.arange(W, device=device).view(1, 1, W).float()

    row_valid = (rows >= y1.view(-1, 1, 1).float()) & \
                (rows <= y2.view(-1, 1, 1).float())
    col_valid = (cols >= x1.view(-1, 1, 1).float()) & \
                (cols <= x2.view(-1, 1, 1).float())

    box_mask = (row_valid & col_valid).float()
    return masks * box_mask

def process_mask_proto(coef, proto, boxes, orig_h, orig_w):
    if coef.shape[0] == 0:
        return np.zeros((0, orig_h, orig_w), dtype=np.uint8)

    C = coef.shape[1]
    ph, pw = proto.shape[1], proto.shape[2]

    proto_flat = proto.reshape(C, -1)
    mask = coef @ proto_flat
    mask = mask.reshape(-1, ph, pw)

    mask_t = torch.from_numpy(mask).unsqueeze(1).float()
    mask_up = F.interpolate(mask_t, size=(orig_h, orig_w),
                            mode='bilinear', align_corners=False)
    mask_up = mask_up.squeeze(1)

    mask_up = crop_mask_to_box_torch(mask_up, boxes)

    masks = (mask_up > 0.5).cpu().numpy().astype(np.uint8)
    return masks

__all__ = [
    'polygons_to_mask',
    'polygons_to_masks_per_gt',
    'crop_mask_to_box_torch',
    'process_mask_proto',
]