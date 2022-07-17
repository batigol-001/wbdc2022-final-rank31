import time
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)

import math
import yaml
import torch
from torch.utils.data import SequentialSampler, DataLoader
from configs.config import parse_args
from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id, category_id_to_lv2id
from models.model import TwoStreamModel

from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex import amp

import random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from dataset.utils import distributed_concat, SequentialDistributedSampler

from torch2trt import torch2trt


def inference():
    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    args = parse_args()
    config = yaml.load(open("configs/Finetune.yaml", 'r'), Loader=yaml.Loader)


    # 1. load data
    #dataset = MultiModalDataset(args, config, args.test_annotation, args.test_zip_frames, data_index=None, test_mode=True)

    # # test
    data_index = range(25000)
    dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, data_index=data_index, test_mode=True)
    batch_size = config["test_batch_size"]
    if rank == 0:
        print("数据量:", len(dataset))
        print("batch_size", batch_size)


    sampler = SequentialDistributedSampler(dataset, batch_size//2)#torch.utils.data.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size//2,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            prefetch_factor=8,
                            num_workers=4)

    # 2. load model
    ckpt_file = args.ckpt_file
    if rank == 0:
        print(f"model_name:{ckpt_file}")
    model = TwoStreamModel(args, config)
    data = [torch.ones(batch_size, args.bert_seq_length, dtype=torch.int32),
            torch.ones(batch_size, args.bert_seq_length, dtype=torch.int32),
            torch.randn(batch_size, config["max_frames"], 3, 224, 224),
            torch.ones(batch_size, config["max_frames"], dtype=torch.int32),
            ]
    model = torch2trt(model, data, fp16_mode=True)

    model.to(device)
    model = amp.initialize(model, opt_level="O1")

    checkpoint = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # model = torch.nn.parallel.DataParallel(model.cuda())
    model = DistributedDataParallel(model, delay_allreduce=True)
    model.eval()



    # 3. inference
    predictions = []
    # labels = []
    s_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            video_feature = batch["frame_input"].to(device, non_blocking=True)
            video_mask = batch['frame_mask'].to(device, non_blocking=True)

            pred_label_id = torch.argmax(model(text_input_ids, text_mask, video_feature, video_mask), 1)
            predictions.append(pred_label_id)

    predictions = distributed_concat(torch.cat(predictions, dim=0), len(dataset))
    predictions = predictions.cpu().numpy()


    if rank == 0:
        print("推理时长", time.time() - s_time)
        print("预测数据量", len(predictions))
        print(predictions)
        # 4. dump results
        with open(args.test_output_csv, 'w') as f:
            for pred_label_id, ann in zip(predictions, dataset.anns):
                video_id = ann['id']
                category_id = lv2id_to_category_id(pred_label_id)
                f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    s_time = time.time()
    inference()
    print("cost time:",time.time() - s_time)
