import argparse
import os
import cv2

from swapfighter import SF
from swapfighter.tasks import list_tasks


def build_argparser():
    p = argparse.ArgumentParser('SWAP-Fighter Inference')
    p.add_argument('--task', type=str, default=None, choices=list_tasks(),
                   help='不填则从权重文件 meta 自动推断')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--output', type=str, default='output.jpg',
                   help='单图输出路径')
    p.add_argument('--output-dir', type=str, default=None,
                   help='多图输出目录（覆盖 --output）')
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])

    # 权重格式
    p.add_argument('--format', type=str, default='pytorch',
                   choices=['pytorch', 'onnx'],
                   help='权重格式：pytorch (.pth) 或 onnx (.onnx)')

    p.add_argument('--conf', type=float, default=None)
    p.add_argument('--nms', type=float, default=None)
    p.add_argument('--img-size', type=int, default=None)

    # 同时支持 --class-name 和 --class-names
    p.add_argument('--class-name', '--class-names',
                   dest='class_names', nargs='+', default=None,
                   help='类别名称列表（空格分隔），会覆盖 meta 中的类名')

    p.add_argument('--top-k', type=int, default=20,
                   help='结果表格显示前 K 个检测框（分类任务为 top-k）')
    p.add_argument('--threshold', type=float, default=None)

    # ---------- 显示相关 ----------
    p.add_argument('--show', action='store_true', default=False,
                   help='推理后弹出窗口显示结果图片（按任意键关闭）')
    p.add_argument('--show-size', type=int, default=1280,
                   help='窗口显示的最大宽度（像素），0 表示原始尺寸')
    p.add_argument('--no-show', action='store_true', default=False,
                   help='(已废弃) 不打开显示窗口')

    return p


def _show_image(img_vis, show_size=1280, window_name='SWAP-Fighter'):
    """弹出窗口显示图片，按任意键关闭。"""
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

        print(f"\n  [Display] 按任意键关闭窗口...")
        cv2.imshow(window_name, img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"\n  [Display] 无法打开显示窗口（无 GUI 环境？）: {e}")


def main():
    args = build_argparser().parse_args()

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

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cfg['output_dir'] = args.output_dir
    else:
        cfg['output_image'] = args.output

    # ============================================================
    # 按 --format 分流
    # ============================================================
    if args.format == 'onnx':
        from swapfighter.engine.predictors.onnx import ONNXPredictor
        pred = ONNXPredictor(cfg, weights=args.weights)
        results = pred.run(args.image)
    else:
        model = SF(weights=args.weights, task=args.task, cfg=cfg)
        results = model.predict(args.image)

    # ---------- 多图统一保存 ----------
    if args.output_dir:
        for r in results:
            vis = r.get('img_vis')
            if vis is None:
                continue
            name = os.path.splitext(os.path.basename(r['image']))[0] + '.jpg'
            vis_path = os.path.join(args.output_dir, name)
            cv2.imwrite(vis_path, vis)

    # ---------- 显示窗口 ----------
    if args.show and not args.no_show:
        for r in results:
            vis = r.get('img_vis')
            if vis is None:
                continue
            title = os.path.basename(r.get('image', 'result'))
            _show_image(vis, show_size=args.show_size,
                        window_name=f'SWAP-Fighter - {title}')


if __name__ == '__main__':
    main()