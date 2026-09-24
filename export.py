import argparse
import os

from swapfighter import SF


def build_argparser():
    p = argparse.ArgumentParser('SWAP-Fighter ONNX Exporter')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--dynamic', action='store_true')
    p.add_argument('--simplify', action='store_true')
    p.add_argument('--opset', type=int, default=11)
    p.add_argument('--end2end', action='store_true')
    p.add_argument('--conf', type=float, default=None)
    p.add_argument('--nms', type=float, default=None)
    p.add_argument('--max-det', type=int, default=300)
    return p


def main():
    args = build_argparser().parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.weights)[0] + '.onnx'

    model = SF(weights=args.weights)
    model.export(
        format='onnx',
        output=args.output,
        batch_size=args.batch_size,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        end2end=args.end2end,
        conf=args.conf,
        nms=args.nms,
        max_det=args.max_det,
    )


if __name__ == '__main__':
    main()