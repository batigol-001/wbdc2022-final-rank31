
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)
# import ruamel_yaml as yaml
import yaml
import time
import torch
import json
import gc
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup

from configs.config import parse_args
from utlils.util import setup_device, setup_seed, build_optimizer, evaluate, setup_logging
from utlils.util import EMA, FGM, PGD
from utlils.category_id_map import category_id_to_lv2id
from dataset.data_helper import create_dataloaders

from models.model import TwoStreamModel

import random
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import multiprocessing as mp
import torch.backends.cudnn as cudnn
gc.enable()

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            # batch = {key: value.cuda() for key, value in batch.items()}
            text_input_ids = batch['text_input_ids'].cuda()
            text_mask = batch['text_attention_mask'].cuda()
            video_feature = batch["frame_input"].cuda()
            video_mask = batch['frame_mask'].cuda()
            loss, _, pred_label_id, label = model(text_input_ids, text_mask, video_feature, video_mask, batch["label"].cuda(), 0.4)
            loss = loss.mean()
            predictions.extend(pred_label_id.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
            losses.append(loss.detach().cpu().numpy())
        loss = sum(losses) / len(losses)
        results = evaluate(predictions, labels)
    model.train()
    return loss, results, predictions, labels

def print_info(args):
    if args.fp16:
        print(" FP16 Starting")

    if args.use_ema:
        print(" EMA Starting")

    if args.use_adv == 1:
        print("FGM Starting")
    elif args.use_adv == 2:
        print("PGD Starting")


from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, ConcatDataset
from dataset.data_helper import MultiModalDataset
from functools import partial

def create_ddp_dataloaders(args, config, train_index, val_index):
    # args.tf_idf_model = load_train_model(args.tf_idf_file)

    if train_index is not None and val_index is not None:

        train_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, train_index)
        val_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, val_index)
        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=config["train_batch_size"],
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = dataloader_class(val_dataset,
                                          batch_size=config["val_batch_size"],
                                          sampler=val_sampler,
                                          drop_last=False)
    else:
        train_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, None)

        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                       prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = RandomSampler(train_dataset, generator=torch.Generator())
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=config["train_batch_size"],
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = None

    return train_dataloader, val_dataloader


def train_and_validate(local_rank, ngpus_per_node, args, config, train_index, val_index, fold_idx):

    args.global_rank = args.node_rank * ngpus_per_node + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.global_world_size,
        rank=args.global_rank,
    )

    args.device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    if args.global_rank == 0:
        print(f"[init] == local rank: {local_rank}, global rank: {args.global_rank} ==")    

    # fix the seed for reproducibility
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    #  load data
    train_dataloader, val_dataloader = create_ddp_dataloaders(args, config, train_index, val_index)

    epochs = config["max_epochs"]
    setps_per_epoch = len(train_dataloader)
    max_steps = epochs * setps_per_epoch
    num_warmup_steps = int(max_steps * config["warmup_ratio"])

    if args.global_rank == 0:
        print(f"total epochs: {epochs}")
        print(f"total steps: {max_steps}")
        print(f"steps per epochs: {setps_per_epoch}")
        print(f"warmup steps: {num_warmup_steps}")
        print_info(args)

    # build model and optimizers
    model = TwoStreamModel(args, config)

    # for n, m in model.named_parameters():
    #     print(n)

    pretrain_file = config["pretrain_file"]
    if pretrain_file and os.path.exists(pretrain_file):
        print(f"加载已经预训练过的模型, file= {pretrain_file}")
        model.load_state_dict(torch.load(pretrain_file, map_location='cpu'), strict=False)


    optimizer = build_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max_steps)
    start_epoch = 1

    # if config['resume'] != "" and os.path.exists(config['resume']):
    #     # 加载之前训练过的模型
    #     args.logger.warning(f"loading model and params from {config['resume']}")
    #     checkpoint = torch.load(config['resume'], map_location='cpu')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.cuda()
    #
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1

    # print(f" start_epoch  = {start_epoch}")

    model = model.to(args.device)

    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()

    torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if args.use_ema:
        ema = EMA(model, 0.999)
        ema.register()


    if args.use_adv == 1:
        fgm = FGM(model, epsilon=0.5, emb_name='word_embeddings.')
    elif args.use_adv == 2:
        pgd = PGD(model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
        K = 3
    #  training
    best_score = 0.0
    for epoch in range(start_epoch, epochs+1):
        start_time = time.time()
        print_loss = 0.0
        ema_loss = 0.0
        ema_acc = 0.0
        print_step = 0

        for i, batch in enumerate(train_dataloader):
            model.train()
            #batch = {key: value.to(args.device) for key, value in batch.items()}
            text_input_ids = batch['text_input_ids'].to(args.device)
            text_mask = batch['text_attention_mask'].to(args.device)
            video_feature = batch["frame_input"].to(args.device)
            video_mask = batch['frame_mask'].to(args.device)
            labels = batch["label"].to(args.device)

            if epoch > 1:
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1, i / len(train_dataloader))

            loss,accuracy, _, _ = model(text_input_ids, text_mask, video_feature, video_mask, labels, alpha)
            loss = loss.mean()
            accuracy = accuracy.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if args.use_adv == 1:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                loss_adv,_, _, _ = model(text_input_ids, text_mask, video_feature, video_mask, alpha)  # 这部分一定要注意model该传入的参数
                loss_adv = loss_adv.mean()
                loss_adv.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                fgm.restore()  # 恢复embedding参数
            elif args.use_adv == 2:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv,_, _, _ = model(text_input_ids, text_mask, video_feature, video_mask, alpha)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                pgd.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema.update()
            optimizer.zero_grad()

            print_step += 1
            print_loss += loss.item()
            ema_loss = 0.9 * ema_loss + 0.1 * loss
            ema_acc = 0.9 * ema_acc + 0.1 * accuracy

            if args.global_rank == 0 and print_step % config["print_steps"] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : loss {print_loss/print_step:.3f}, "
                                 f"ema_acc {ema_acc:.3f}, learning_rate: {lr}")

            # 评估
            if args.use_ema:
                ema.apply_shadow()

            if val_dataloader is not None:
                if epoch >= 3 and print_step % 500 == 0:
                    s_time = time.time()
                    loss, results, predictions, labels = validate(model, val_dataloader)
                    results = {k: round(v, 4) for k, v in results.items()}
                    if args.global_rank == 0:
                        print(f"Eval epoch {epoch} step [{print_step} / {setps_per_epoch}]  : loss {loss:.3f}, {results},"
                                         f"cost time {time.time() - s_time} ")
                        score = results["mean_f1"]
                        if score > best_score:
                            best_score = score
                            save_file = f'{args.savedmodel_path}/model_fold_{fold_idx}_best_score.bin'
                            model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                            torch.save({"model_state_dict": model_save}, save_file)
                            print(f"save mode to file {save_file}")
            else:
                # 全量保存最后一轮
                if args.global_rank == 0 and epoch == 3 and print_step % 500 == 0:
                    print(f"全量训练无评估")
                    model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    save_file = f'{args.savedmodel_path}/model_fold_{fold_idx}_epoch_{epoch}_step_{print_step}.bin'
                    torch.save({"model_state_dict": model_save}, save_file)
                    print(f"save mode to file {save_file}")

            # 结束评估
            if args.use_ema:
                ema.restore()
        # epoch end
        if args.global_rank == 0:
            print(f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : loss {print_loss / print_step:.3f}, "
                         f"ema_acc {ema_acc:.3f}")

        # 评估
        if args.use_ema:
            ema.apply_shadow()

        if val_dataloader is not None:
            s_time = time.time()
            loss, results, predictions, labels = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            if args.global_rank == 0:
                print(f"Eval epoch {epoch} : loss {loss:.3f}, {results} , cost time {time.time() - s_time} ")
                score = results["mean_f1"]
                if score > best_score:
                    best_score = score
                    save_file = f'{args.savedmodel_path}/model_fold_{fold_idx}_best_score.bin'
                    model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save({"model_state_dict": model_save}, save_file)
                    print(f"save mode to file {save_file}")

        else:
            # 全量保存最后一轮
            if args.global_rank == 0:
                print(f"全量训练无评估")
                # if epoch == 3:
                model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                save_file = f'{args.savedmodel_path}/model_fold_{fold_idx}_epoch_{epoch}_step_{print_step}.bin'
                torch.save({"model_state_dict": model_save }, save_file)
                print(f"save mode to file {save_file}")
            # 结束评估
        if args.use_ema:
            ema.restore()
        dist.barrier()
        if args.global_rank == 0:
            print(f"cost time: {time.time() - start_time}")
        # # todo
        # if epoch == 3:
        #     break

    print(f" train finished!..................")
    torch.cuda.empty_cache()

def main():

    args = parse_args()

    config = yaml.load(open("configs/Finetune.yaml", 'r'), Loader=yaml.Loader)
    config["lr"] = float(config["lr"])
    config["other_lr"] = float(config["other_lr"])

    version = str(config["version"])
    log_path = os.path.join(args.log_path, "finetune")
    filename = os.path.join(log_path, f"finetune_{version}.log")

    args.logger = setup_logging(log_path, filename)


    args.savedmodel_path = os.path.join(args.model_path, "finetune", version)
    os.makedirs(args.savedmodel_path, exist_ok=True)

    print("Training/evaluation parameters: %s, %s", args, config)

    yaml.dump(config, open(os.path.join(args.savedmodel_path, 'config.yaml'), 'w'))

    args.global_world_size = args.ngpus_per_node * args.nodes
    # 判断是否多折
    if args.n_splits > 0:
        with open(args.train_annotation, 'r', encoding='utf8') as f:
            anns = json.load(f)

        X = range(len(anns))[:]
        y = [category_id_to_lv2id(c["category_id"]) for c in anns][:]
        kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for fold_idx, (train_index, val_index) in enumerate(kfold.split(X, y)):

            print(f"-----------------开始训练, 第 {fold_idx+1} 折------------------------")
            print(f"train data num = {len(train_index)}, val data num = {len(val_index)}")
            #train_and_validate(args, config, train_index, val_index, fold_idx+1)
            mp.spawn(train_and_validate, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args, config, train_index, val_index, fold_idx+1))
            print(f"-----------------结束训练, 第 {fold_idx+1} 折------------------------")
            # todo 是否开启
            if fold_idx+1 == 1:
                break
    # 全量
    else:
        train_and_validate(args, config, None, None, 0)


if __name__ == '__main__':
    main()

