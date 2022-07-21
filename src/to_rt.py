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
batch_size = 2#config["test_batch_size"]
ckpt_file = args.ckpt_file
print(f"model_name:{ckpt_file}")

model = TwoStreamModel(args, config)
checkpoint = torch.load(ckpt_file, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

text_input_ids = torch.zeros(batch_size//2, args.bert_seq_lenght, dtype=torch.long)
text_mask = torch.zeros(batch_size//2, args.bert_seq_lenght, dtype=torch.long)
video_feature = torch.zeros(batch_size//2, config["max_frames"], 3, 224, 224, dtype=torch.float32)
video_mask = torch.zeros(batch_size//2, config["max_frames"], dtype=torch.long)
input_names = ["text_input_ids", "text_mask", "video_feature", "video_mask"]
output_names = ["predictions"]


torch.onnx.export(model, (text_input_ids, text_mask, video_feature, video_mask), 'model.onnx', verbose=False, input_names=input_names,
                  output_names=output_names,
                  do_constant_folding=True)




