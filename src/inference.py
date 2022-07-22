import time
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
#import ruamel_yaml as yaml
import yaml
import torch
from torch.utils.data import SequentialSampler, DataLoader
from configs.config import parse_args
from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id
from models.model import TwoStreamModel


def inference():
    print("cuda: ", torch.cuda.is_available())
    args = parse_args()
    config = yaml.load(open("configs/Finetune.yaml", 'r'), Loader=yaml.Loader)


    # 1. load data
    # dataset = MultiModalDataset(args, config, args.test_annotation, args.test_zip_frames, data_index=None, test_mode=True)

    # # test
    data_index = range(25000)
    dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, data_index=data_index, test_mode=True)

    print("数据量:", len(dataset))
    batch_size = config["test_batch_size"]
    print("batch_size", batch_size)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            prefetch_factor=8,
                            num_workers=4)

    # 2. load model
    ckpt_file = args.ckpt_file
    print(f"model_name:{ckpt_file}")
    model = TwoStreamModel(args, config)
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = torch.nn.parallel.DataParallel(model.half().cuda())
    model.eval()

    # 3. inference
    predictions = []
    s_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            text_input_ids = batch['text_input_ids'].int().cuda()
            text_mask = batch['text_attention_mask'].int().cuda()
            video_feature = batch["frame_input"].half().cuda()
            video_mask = batch['frame_mask'].int().cuda()


            pred_label_id = torch.argmax(model(text_input_ids, text_mask, video_feature, video_mask), 1)
            predictions.extend(pred_label_id.cpu().numpy())
    print(time.time() - s_time)
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
