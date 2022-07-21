import torch
import time
import logging
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id

logger = logging.getLogger(__name__)
DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            output_shape = engine.get_binding_shape(binding)
            if len(output_shape) == 3:
                dims = trt.DimsCHW(engine.get_binding_shape(binding))
                output_shape = (dims.c, dims.h, dims.w)
            elif len(output_shape) == 2:
                dims = trt.Dims2(output_shape)
                output_shape = (dims[0], dims[1])
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

    return inputs, outputs, bindings, stream


def do_inference(batch, context, bindings, inputs, outputs, stream):
    assert len(inputs) == 3
    inputs[0].host = np.ascontiguousarray(batch['frame_input'], dtype=np.float32)
    inputs[1].host = np.ascontiguousarray(batch['title_input'], dtype=np.int32)
    inputs[2].host = np.ascontiguousarray(batch['title_mask'], dtype=np.int32)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    outputs_dict = {}
    for out in outputs:
        outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
    return outputs_dict


def load_tensorrt_engine(filename):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def inference():
    args = parse_args()

    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=10,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    predictions = []
    with load_tensorrt_engine('model.trt.engine') as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            for data in dataloader:
                batch = {k: v.numpy() for k, v in data.items()}
                outputs_dict = do_inference(batch, context, bindings, inputs, outputs, stream)
                predictions.extend(np.argmax(outputs_dict['output_0'], axis=1))

    # 3. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
