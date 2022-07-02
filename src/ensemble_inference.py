import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader

from single_stream.config import parse_args
from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id

from single_stream.wx_challenge import WXChallengeModel
from models.two_stream_model import TwoStreamModel


def ensemble_inference():
    args = parse_args()
    print("test_annotation:", args.test_annotation)
    print("test_zip_feats:", args.test_zip_feats)
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, data_index=None, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    test_data_num = len(dataset)
    avg_preds = np.zeros((test_data_num, 200))
    print("text nums:", test_data_num)

    # # swa model
    # swa_avg_preds = np.zeros((test_data_num, 200))
    # ckpt_files = ["data/models/single_stream/finetune/2/model_fold_0_epoch_3_step_3125.bin",
    #               "data/models/two_stream/finetune/2/model_fold_0_epoch_3_step_3125.bin"]
    # print("all data model:", ckpt_files)
    # for idx, _ckpt_file in enumerate(ckpt_files):
    #     args.logger.info(f"model_name:{_ckpt_file}")
    #     if idx == 0:
    #         model = WXChallengeModel(args, task=["tag"])
    #     else:
    #         model = TwoStreamModel(args, task=["tag"])
    #     checkpoint = torch.load(_ckpt_file, map_location='cpu')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     model = model.cuda()
    #     model.eval()
    #
    #     # 推理
    #     preds = np.zeros((test_data_num, 200))
    #     batch_size = args.test_batch_size
    #     with torch.no_grad():
    #         idx = 0
    #         for batch in dataloader:
    #             # print(1)
    #             batch = {key: value.cuda() for key, value in batch.items()}
    #             pred = model(batch, inference=True)
    #             preds[idx * batch_size: batch_size * (idx + 1)] = pred.detach().cpu().numpy()
    #             idx += 1
    #
    #     swa_avg_preds += preds

    # 2.单流
    ckpt_path = "data/models/single_stream/finetune/1/"
    ckpt_files = [ckpt_path + f"model_fold_{i}_best_score.bin" for i in range(1, 11)]
    # ckpt_files = ["/root/autodl-tmp/models/single_stream/finetune/12/swa_model.bin"]
    if ckpt_files:
        print("加载单流模型")
    print(ckpt_files)
    for _ckpt_file in ckpt_files:
        args.logger.info(f"model_name:{_ckpt_file}")
        model = WXChallengeModel(args, task=["tag"])
        checkpoint = torch.load(_ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()

        # 推理
        preds = np.zeros((test_data_num, 200))
        batch_size = args.test_batch_size
        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                # print(1)
                batch = {key: value.cuda() for key, value in batch.items()}
                pred = model(batch, inference=True)
                preds[idx*batch_size: batch_size*(idx+1)] = pred.detach().cpu().numpy()
                idx += 1

        avg_preds += preds

    # 双流
    ckpt_path = "data/models/two_stream/finetune/1/"
    ckpt_files = [ckpt_path + f"model_fold_{i}_best_score.bin" for i in range(1, 11)]
    # ckpt_files = ["/root/autodl-tmp/models/two_stream/finetune/v2_7/swa_model.bin"]
    if ckpt_files:
        print("加载双流模型")
    print(ckpt_files)
    for _ckpt_file in ckpt_files:
        args.logger.info(f"model_name:{_ckpt_file}")
        model = TwoStreamModel(args, task=["tag"])
        checkpoint = torch.load(_ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()

        # 推理
        preds = np.zeros((test_data_num, 200))
        batch_size = args.test_batch_size
        with torch.no_grad():
            idx = 0
            for batch in dataloader:
                # print(1)
                batch = {key: value.cuda() for key, value in batch.items()}
                pred = model(batch, inference=True)
                preds[idx * batch_size: batch_size * (idx + 1)] = pred.detach().cpu().numpy()
                idx += 1

        avg_preds += preds

    # avg_preds = avg_preds * 0.7 + swa_avg_preds * 10 * 0.3

    predictions = np.argmax(avg_preds, 1)
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')

if __name__ == '__main__':
    ensemble_inference()
