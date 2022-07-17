import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
import time
#import ruamel_yaml as yaml
import yaml

import torch
from torch.utils.data import SequentialSampler, DataLoader
from configs.config import parse_args
from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id
from models.model import TwoStreamModel



args = parse_args()
config = yaml.load(open("configs/Finetune.yaml", 'r'), Loader=yaml.Loader)
batch_size = config["test_batch_size"]
ckpt_file = args.ckpt_file
print(f"model_name:{ckpt_file}")

model = TwoStreamModel(args, config)
checkpoint = torch.load(ckpt_file, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

data = (torch.ones(batch_size, args.bert_seq_length, dtype=torch.int32),
        torch.ones(batch_size, args.bert_seq_length, dtype=torch.int32),
        torch.randn(batch_size, config["max_frames"], 3, 224, 224),
        torch.ones(batch_size, config["max_frames"], dtype=torch.int32),
        )

torch.onnx.export(model, data, 'model.onnx', verbose=False, opset_version=12, input_names=["input_0"],
                  output_names=["output_0"],
                  do_constant_folding=True)




