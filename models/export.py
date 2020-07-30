"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img-size 640 --batch-size 1 --device cuda --include-detect-layer
"""

import argparse

from models.common import *
from utils import google_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path, defaults to ./yolov5s.pt')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size, defaults to [640, 640]')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size, defaults to 1')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu, defaults to cuda')
    parser.add_argument('--include-detect-layer', action='store_true', help='include Detect() layer in exported models')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Device
    map_location = torch.device(opt.device) if opt.device else None
    if not map_location:
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size)).to(map_location)  # image size(1,3,320,192) iDetection

    # Remove/include Detect() layer by setting export to True/False respectively.
    # The Detect() layer combines the results from multiple output layers into one set of detections. 
    # This work needs to be done during postprocessing if not included as a layer in the export.
    # See https://github.com/ultralytics/yolov5/issues/343#issuecomment-658021043 for more details.
    detect_layer_export = not opt.include_detect_layer

    # Load PyTorch model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=map_location)['model'].float()
    model.eval()
    model.model[-1].export = detect_layer_export
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img)
        ts.save(f)
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
