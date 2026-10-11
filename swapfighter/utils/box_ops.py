import math
import cv2
import numpy as np
import torch

LOG_SCALE_MIN = -4.0
LOG_SCALE_MAX = 6.0
MIN_POS_PER_GT = 4

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top), shape

def cxcywh_angle_to_obb(cx, cy, w, h, angle):
    N = cx.shape[0]
    device = cx.device
    half_w = w / 2
    half_h = h / 2

    corners = torch.zeros((N, 4, 2), device=device, dtype=cx.dtype)
    corners[:, 0, 0] = half_w; corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w; corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w; corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w; corners[:, 3, 1] = -half_h

    cos_a = torch.cos(angle); sin_a = torch.sin(angle)
    rot_matrix = torch.stack([
        torch.stack([cos_a, sin_a], dim=1),
        torch.stack([-sin_a, cos_a], dim=1)
    ], dim=1)

    corners_rot = torch.matmul(corners, rot_matrix)
    center = torch.stack([cx, cy], dim=1).unsqueeze(1)
    corners_final = corners_rot + center
    obboxes = corners_final.reshape(N, 8)
    return torch.clamp(obboxes, 0, 1)

def obb_xy8_to_cxcywh_angle_np(obboxes_8):
    pts = obboxes_8.reshape(-1, 4, 2).astype(np.float32)
    cx = pts[:, :, 0].mean(axis=1)
    cy = pts[:, :, 1].mean(axis=1)

    v1 = 0.5 * ((pts[:, 1] - pts[:, 0]) + (pts[:, 2] - pts[:, 3]))
    v2 = 0.5 * ((pts[:, 2] - pts[:, 1]) + (pts[:, 3] - pts[:, 0]))
    w1 = np.linalg.norm(v1, axis=1)
    h1 = np.linalg.norm(v2, axis=1)
    mask = w1 >= h1
    w = np.where(mask, w1, h1)
    h = np.where(mask, h1, w1)
    angle = np.where(mask,
                     np.arctan2(v1[:, 1], v1[:, 0]),
                     np.arctan2(v2[:, 1], v2[:, 0]))
    return np.stack([cx, cy, w, h, angle], axis=1).astype(np.float32)

def cxcywh_angle_to_obb_np(cx, cy, w, h, angle):
    N = cx.shape[0]
    half_w = w / 2
    half_h = h / 2

    corners = np.zeros((N, 4, 2), dtype=np.float32)
    corners[:, 0, 0] = half_w; corners[:, 0, 1] = -half_h
    corners[:, 1, 0] = half_w; corners[:, 1, 1] = half_h
    corners[:, 2, 0] = -half_w; corners[:, 2, 1] = half_h
    corners[:, 3, 0] = -half_w; corners[:, 3, 1] = -half_h

    cos_a = np.cos(angle); sin_a = np.sin(angle)
    rot = np.stack([
        np.stack([cos_a, sin_a], axis=1),
        np.stack([-sin_a, cos_a], axis=1)
    ], axis=1)

    corners_rot = np.matmul(corners, rot)
    center = np.stack([cx, cy], axis=1)[:, None, :]
    corners_final = corners_rot + center
    return corners_final.reshape(N, 8).astype(np.float32)