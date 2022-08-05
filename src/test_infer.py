import time
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
#import ruamel_yaml as yaml
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler, DataLoader
from sklearn.model_selection import StratifiedKFold

from configs.config import parse_args
from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id, category_id_to_lv2id
from utlils.util import evaluate
from models.model import TwoStreamModel


def log_softmax(x):
    exp_x = np.exp(x)
    exp_x /= np.sum(exp_x, axis=1, keepdims=True)
    return  exp_x

def inference():
    print("cuda: ", torch.cuda.is_available())
    args = parse_args()
    config = yaml.load(open("configs/Finetune.yaml", 'r'), Loader=yaml.Loader)

    # 加载验证数据
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)

    X = range(len(anns))[:]
    y = [category_id_to_lv2id(c["category_id"]) for c in anns][:]
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold_idx, (train_index, val_index) in enumerate(kfold.split(X, y)):
        # train_index, val_index
        break

    prior = np.zeros(200)
    for idx in train_index:
        label = category_id_to_lv2id(anns[idx]['category_id'])
        prior[label] += 1
    prior/= prior.sum()

    print("训练集先验概率分布:", prior)


    val_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, data_index=val_index,
                                test_mode=False)

    print("验证集数据量:", len(val_dataset))
    batch_size = config["test_batch_size"]
    print("batch_size", batch_size)
    sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            prefetch_factor=8,
                            num_workers=4)


    # # 2. load model
    # ckpt_file = args.ckpt_file
    # print(f"model_name:{ckpt_file}")
    # model = TwoStreamModel(args, config)
    # checkpoint = torch.load(ckpt_file, map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
    #
    # model = torch.nn.parallel.DataParallel(model.half().cuda())
    # model.eval()

    # # 3. inference
    # pred_proabs = np.zeros((len(val_index), 200))
    # predictions = []
    # labels = []
    # s_time = time.time()
    # with torch.no_grad():
    #     for i, batch in enumerate(val_dataloader):
    #         text_input_ids = batch['text_input_ids'].int().cuda()
    #         text_mask = batch['text_attention_mask'].int().cuda()
    #         video_feature = batch["frame_input"].half().cuda()
    #         video_mask = batch['frame_mask'].int().cuda()
    #
    #         pred_proab = model(text_input_ids, text_mask, video_feature, video_mask)
    #         pred_label_id = torch.argmax(pred_proab, 1)
    #         predictions.extend(pred_label_id.cpu().numpy())
    #         labels.extend(batch["label"].squeeze(dim=1).numpy())
    #
    #         pred_proabs[i*batch_size: (i+1)*batch_size] = pred_proab.cpu().numpy()
    #
    #
    # pred_df = pd.DataFrame(pred_proabs, columns=[f"pred_{i}" for i in range(200)])
    # pred_df["label"] = labels
    # pred_df.to_csv("valid_pred.csv", index=False)

    #
    pred_df = pd.read_csv("valid_pred.csv", index_col=False)
    pred_proabs = pred_df.drop("label", axis=1).values
    labels = pred_df["label"].values
    predictions = np.argmax(pred_proabs, 1)


    results = evaluate(predictions, labels)
    print("total ori result: ",results)
    print(time.time() - s_time)

    # 4. 后处理
    #  评价每个预测结果的不确定性
    k = 3
    pred_proabs = log_softmax(pred_proabs)
    print(pred_proabs)
    pred_topk = np.sort(pred_proabs, axis=1)[:, -k:]
    pred_topk /= pred_topk.sum(axis=1, keepdims=True)
    pred_uncertainty = -(pred_topk * np.log(pred_topk)).sum(axis=1) / np.log(k)
    #  选择阈值，划分高、低置信度两部分
    threshold = 0.8
    preds_confident = pred_proabs[pred_uncertainty < threshold]
    preds_unconfident = pred_proabs[pred_uncertainty >= threshold]

    trues_confident = labels[pred_uncertainty < threshold]
    trues_unconfident = labels[pred_uncertainty >= threshold]

    results_confident = evaluate(np.argmax(preds_confident, 1), trues_confident)
    results_unconfident = evaluate(np.argmax(preds_unconfident, 1), trues_unconfident)
    print("ori condifent nums", len(preds_confident))
    print("ori condifent results:", results_confident)
    print("ori uncondifent nums", len(preds_unconfident))
    print("ori uncondifent results:", results_unconfident)


    # 逐个修改低置信度样本，并重新评价准确率
    # new_preds_unconfident = np.zeros(preds_unconfident.shape)
    def modify_un_confident(y):
        right, alpha, iters = 0, 1, 1
        # for i, y in enumerate(preds_unconfident):
        Y = np.concatenate([preds_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y**alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        # new_preds_unconfident[i] = y
        return y


    new_preds_unconfident =  np.apply_along_axis(modify_un_confident, 0, preds_unconfident)

    results_unconfident = evaluate(np.argmax(new_preds_unconfident, 1), trues_unconfident)
    print("new uncondifent results:", results_unconfident)

    new_preds = np.concatenate((preds_confident, new_preds_unconfident))
    labels = np.concatenate((trues_confident, trues_unconfident))

    new_results = evaluate(np.argmax(new_preds, 1), labels)
    print("total new result: ", new_results)
    # 输出修正后的准确率

if __name__ == '__main__':
    s_time = time.time()
    inference()
    print("cost time:",time.time() - s_time)
