# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import inspect
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')
try:
    from fvcore.nn import parameter_count_table
except ImportError:
    raise ImportError('Please install fvcore to use parameter counting.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument(
        '--param-shapes',
        type=int,
        nargs='+',
        action='append',
        default=None,
        help='input image sizes for parameter reporting; can be repeated, '
        'e.g. --param-shapes 512 512 --param-shapes 1024 512')
    parser.add_argument(
        '--param-depth',
        type=int,
        default=3,
        help='max module depth for fvcore parameter table')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def _normalize_shapes(shapes, fallback):
    if shapes is None:
        shapes = [fallback]
    normalized = []
    for shape in shapes:
        if len(shape) == 1:
            normalized.append((shape[0], shape[0]))
        elif len(shape) == 2:
            normalized.append(tuple(shape))
        else:
            raise ValueError('invalid input shape')
    return normalized


def _build_dummy_extras(model: BaseSegmentor, input_shape: tuple[int, int, int]):
    extra_names = []
    if hasattr(model, 'extra_names') and model.extra_names is not None:
        extra_names = list(model.extra_names)
    if hasattr(model, 'key_sec') and isinstance(model.key_sec, str):
        extra_names.append(model.key_sec)

    extra_names = [name for name in extra_names if name != 'img']
    if not extra_names:
        return {}

    _, height, width = input_shape
    extras = {}
    for name in set(extra_names):
        if name == 'dct':
            extras[name] = torch.rand(1, height, width)
            extras.setdefault('qtable', torch.ones(1, 8, 8))
        elif name == 'qtable':
            extras[name] = torch.ones(1, 8, 8)
        else:
            extras[name] = torch.rand(3, height, width)
    return extras


def _preprocessor_supports_extras(model: BaseSegmentor) -> bool:
    data_preprocessor = getattr(model, 'data_preprocessor', None)
    return data_preprocessor is not None and hasattr(data_preprocessor, 'extra_pad_val')


def _model_needs_extras(model: BaseSegmentor) -> bool:
    signature = inspect.signature(model.forward)
    return 'extras' in signature.parameters


def inference(args: argparse.Namespace,
              logger: MMLogger) -> tuple[dict, BaseSegmentor, tuple]:
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    if _model_needs_extras(model) or _preprocessor_supports_extras(model):
        dummy_extras = _build_dummy_extras(model, input_shape)
        data_batch['extra'] = {key: [value] for key, value in dummy_extras.items()}
    data = model.data_preprocessor(data_batch)
    model.eval()
    if hasattr(model, 'vis_preprocessor'):
        model.vis_preprocessor = None
    if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
        # TODO: Support MaskFormer and Mask2Former
        raise NotImplementedError('MaskFormer and Mask2Former are not '
                                  'supported yet.')
    if _model_needs_extras(model):
        model_inputs = (data['inputs'], data.get('extras', None), data['data_samples'])
    else:
        model_inputs = (data['inputs'], data['data_samples'])
    outputs = get_model_complexity_info(
        model,
        input_shape,
        inputs=model_inputs,
        show_table=False,
        show_arch=False)
    result['flops'] = _format_size(outputs['flops'])
    result['params'] = _format_size(outputs['params'])
    result['compute_type'] = 'direct: randomly generate a picture'
    return result, model, model_inputs


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result, model, _ = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    param_shapes = _normalize_shapes(args.param_shapes, args.shape)
    param_table = parameter_count_table(model, max_depth=args.param_depth)
    total_params = sum(p.numel() for p in model.parameters())
    total_params_fmt = _format_size(total_params)

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    for shape in param_shapes:
        print(f'{split_line}\nParameter stats (fvcore)\n'
              f'Input shape: {shape}\nTotal params: {total_params_fmt}\n'
              f'{split_line}')
        print(param_table)
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()
