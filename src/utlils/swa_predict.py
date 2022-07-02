import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
import copy
import torch
from torch.utils.data import SequentialSampler, DataLoader
from configs.config import parse_args
from models.two_stream_model import TwoStreamModel

from dataset.data_helper import MultiModalDataset
from utlils.category_id_map import lv2id_to_category_id
from utlils.util import evaluate


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {key: value.cuda() for key,value in batch.items()}
            loss, _, _,pred_label_id, label = model(batch, task=["tag"])
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def swa_interface():
    args = parse_args()

    ckpt_path = "/root/autodl-tmp/models/two_stream/finetune/v2_7/"

    steps = [1500, 2000, 2500, 3000, 3125]

    ckpt_file_lst = [
                       ckpt_path + f"model_fold_0_epoch_3_step_{step}.bin" for step in steps
                     ]
    print(ckpt_file_lst)
    model = TwoStreamModel(args, task=["tag"])
    swa_model = copy.deepcopy(model)
    swa_n = 0
    with torch.no_grad():
        for ckpt_file in ckpt_file_lst:
            args.logger.info(f"model_name:{ckpt_file}")
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1


    swa_model = swa_model.cuda()
    args.logger.info(f"Eval swa model:")

    model_save = swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict()
    save_file = ckpt_path + "swa_model.bin"
    torch.save({"model_state_dict": model_save}, save_file)
    args.logger.info(f"save mode to file {save_file}")


    args.logger.info(f"interface。。。。。。。。。。。")
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers)
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.cuda() for key, value in batch.items()}
            pred_label_id = torch.argmax(swa_model(batch, inference=True),dim=1)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    # swa_eval()
    swa_interface()