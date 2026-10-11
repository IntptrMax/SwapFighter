import argparse
import os
import cv2

from swapfighter import SF
from swapfighter.tasks import list_tasks

def build_argparser():
    p = argparse.ArgumentParser('SWAP-Fighter Inference')
    p.add_argument('--task', type=str, default=None, choices=list_tasks(),
                   help='不填则从权重文件 meta 自动推断')
    p.add_argument('--weights', type=str, required=True,
                   help='权重路径：.pth/.pt → PyTorch，.onnx → ONNX')
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--output', type=str, default='output.jpg')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])

    p.add_argument('--format', type=str, default=None,
                   choices=['pytorch', 'onnx'],
                   help='推理框架。不填则根据 --weights 后缀自动判断：'
                        '.onnx → onnx，.pth/.pt → pytorch')

    p.add_argument('--conf', type=float, default=None)
    p.add_argument('--nms', type=float, default=None)
    p.add_argument('--img-size', type=int, default=None)

    p.add_argument('--class-name', '--class-names',
                   dest='class_names', nargs='+', default=None)

    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--threshold', type=float, default=None)

    p.add_argument('--no-mask', dest='draw_mask',
                   action='store_false', default=None,
                   help='推理 segment 时不画 mask')
    p.add_argument('--mask-alpha', type=float, default=None,
                   help='mask 叠加透明度（0~1，默认 0.5）')

    p.add_argument('--kpt-radius', type=int, default=None,
                   help='关键点可视化半径（默认 6）')
    p.add_argument('--kpt-thr', type=float, default=None,
                   help='关键点显示阈值（默认 0.3）')
    p.add_argument('--no-kpt', dest='draw_kpt',
                   action='store_false', default=None,
                   help='推理 keypoint 时不画关键点')

    p.add_argument('--show', action='store_true', default=False)
    p.add_argument('--show-size', type=int, default=1280)

    return p

def _resolve_format(args):
    weights = args.weights
    ext = os.path.splitext(weights)[1].lower()

    if ext == '.onnx':
        inferred = 'onnx'
    elif ext in ('.pth', '.pt', '.ckpt'):
        inferred = 'pytorch'
    else:
        inferred = None

    if args.format is not None:
        if inferred is not None and inferred != args.format:
            print(f"\n  [WARN] --format={args.format!r} 与权重后缀 "
                  f"{ext!r} 不匹配，仍按 --format 执行。")
        return args.format

    if inferred is None:
        print(f"\n  [WARN] 无法从后缀 {ext!r} 推断推理格式，"
              f"回退到 pytorch。可用 --format 显式指定。")
        return 'pytorch'

    return inferred

def _show_image(img_vis, show_size=1280, window_name='SWAP-Fighter'):
    try:
        if show_size and show_size > 0:
            h, w = img_vis.shape[:2]
            if w > show_size:
                display_w = show_size
                display_h = int(display_w * h / w)
                img_show = cv2.resize(img_vis, (display_w, display_h),
                                      interpolation=cv2.INTER_LINEAR)
            else:
                img_show = img_vis
        else:
            img_show = img_vis
        print("\n  [Display] 按任意键关闭窗口...")
        cv2.imshow(window_name, img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"\n  [Display] 无法打开显示窗口: {e}")

def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重文件不存在: {args.weights}")

    args.format = _resolve_format(args)

    cfg = {
        'device': args.device,
        'top_k': args.top_k,
    }
    if args.conf is not None:
        cfg['conf_threshold'] = args.conf
    if args.nms is not None:
        cfg['nms_threshold'] = args.nms
        cfg['obb_nms_threshold'] = args.nms
    if args.img_size is not None:
        cfg['img_size'] = args.img_size
    if args.threshold is not None:
        cfg['eval_threshold'] = args.threshold
    if args.class_names:
        cfg['class_names'] = list(args.class_names)

    if args.draw_mask is not None:
        cfg['draw_mask'] = args.draw_mask
    if args.mask_alpha is not None:
        cfg['mask_alpha'] = args.mask_alpha

    if args.kpt_radius is not None:
        cfg['kpt_radius'] = args.kpt_radius
    if args.kpt_thr is not None:
        cfg['kpt_thr'] = args.kpt_thr
    if args.draw_kpt is not None:
        cfg['draw_kpt'] = args.draw_kpt

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cfg['output_dir'] = args.output_dir
    else:
        cfg['output_image'] = args.output

    if args.format == 'onnx':
        from swapfighter.engine.predictors.onnx import ONNXPredictor
        pred = ONNXPredictor(cfg, weights=args.weights)
        results = pred.run(args.image)
    else:
        model = SF(weights=args.weights, task=args.task, cfg=cfg)
        results = model.predict(args.image)

    if args.output_dir:
        for r in results:
            vis = r.get('img_vis')
            if vis is None:
                continue
            name = os.path.splitext(os.path.basename(r['image']))[0] + '.jpg'
            cv2.imwrite(os.path.join(args.output_dir, name), vis)

    if args.show:
        for r in results:
            vis = r.get('img_vis')
            if vis is None:
                continue
            title = os.path.basename(r.get('image', 'result'))
            _show_image(vis, show_size=args.show_size,
                        window_name=f'SWAP-Fighter - {title}')

if __name__ == '__main__':
    main()