import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)

# import ruamel_yaml as yaml
import yaml
import time
import numpy as np
import random

import torch
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
from torch import multiprocessing as mp
import torch.backends.cudnn as cudnn

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from functools import partial
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex import amp

from configs.config import parse_args
from utlils.util import setup_device, setup_seed, build_optimizer, setup_logging
from utlils.util import init_distributed_mode, get_rank, get_world_size, is_main_process
from dataset.data_helper import MultiModalDataset
from models.model_pretrain_simple import TwoStreamModel
from dataset.utils import distributed_concat, reduce_tensor

import gc

import warnings
warnings.filterwarnings('ignore')

gc.enable()


def validate(model, val_dataloader):
    model.eval()
    losses = []
    mlm_losses = []
    ita_losses = []
    itm_losses = []

    with torch.no_grad():
        for batch in val_dataloader:
            # batch = {key: value.cuda() for key, value in batch.items()}
            text_input_ids = batch['text_input_ids'].cuda()
            text_mask = batch['text_attention_mask'].cuda()
            video_feature = batch["frame_input"].cuda()
            video_mask = batch['frame_mask'].cuda()

            loss, _, pred_label_id, label = model(text_input_ids, text_mask, video_feature, video_mask, 0.4)

            loss = loss.mean()
            mlm_loss = mlm_loss.mean()
            ita_loss = ita_loss.mean()
            itm_loss = itm_loss.mean()
            losses.append(loss.cpu().numpy())
            mlm_losses.append(mlm_loss.cpu().numpy())
            ita_losses.append(ita_loss.cpu().numpy())
            itm_losses.append(itm_loss.cpu().numpy())
        loss = sum(losses) / len(losses)
        mlm_loss = sum(mlm_losses) / len(mlm_losses)
        ita_loss = sum(ita_losses) / len(ita_losses)
        itm_loss = sum(itm_losses) / len(itm_losses)
    return loss, (mlm_loss, ita_loss, itm_loss)


def create_pretrain_dataloaders(args,  config, annotations, zip_frames):
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    dataset = None
    for idx, (_annotation, _zip_feat) in enumerate(zip(annotations, zip_frames)):

        # index = list(range(500))
        sub_dataset = MultiModalDataset(args, config, _annotation, _zip_feat, None, test_mode=True)
        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = ConcatDataset([dataset, sub_dataset])

    train_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
    train_dataloader = dataloader_class(dataset,
                                    batch_size=config["train_batch_size"],
                                    sampler=train_sampler,
                                    drop_last=True)

    return train_dataloader


def train_worker(rank, local_rank, device,args, config):
    #  load data
    annotations = [args.unlabeled_annotation, args.train_annotation]
    zip_feats = [args.unlabeled_zip_frames, args.train_zip_frames]

    # annotations = [args.train_annotation]
    # zip_feats = [args.train_zip_frames]

    args.logger.info(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # fix the seed for reproducibility
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True



    train_dataloader = create_pretrain_dataloaders(args, config, annotations, zip_feats)


    epochs = config["max_epochs"]
    setps_per_epoch = len(train_dataloader)
    max_steps = epochs * setps_per_epoch
    num_warmup_steps = int(max_steps * config["warmup_ratio"])
    config["print_steps"] = config["print_steps"] * config["accum_step"]

    if rank == 0:
        args.logger.info(f"total epochs: {epochs}")
        args.logger.info(f"total steps: {max_steps}")
        args.logger.info(f"steps per epochs: {setps_per_epoch}")
        args.logger.info(f"warmup steps: {num_warmup_steps}")

    model = TwoStreamModel(args, config)
    # for n, m in model.named_parameters():
    #     print(n)


    optimizer = build_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps//config["accum_step"],
                                                num_training_steps=max_steps//config["accum_step"])




    # 同步BN
    model = convert_syncbn_model(model)
    model.to(device)
    # 混合精度
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    exist_pretrain_file = ""
    if os.path.exists(exist_pretrain_file):
        args.logger.info(f"加载已经预训练过的模型, file= {exist_pretrain_file}")
        checkpoint = torch.load(exist_pretrain_file, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        amp.load_state_dict(checkpoint['amp_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.cuda()
    else:
        start_epoch = 1

        # DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model = DistributedDataParallel(model, delay_allreduce=True)  # device_ids=[local_rank], output_device=local_rank)

    args.logger.info(f" start_epoch  = {start_epoch}")
    dist.barrier()
    for epoch in range(start_epoch, epochs+1):
        start_time = time.time()
        print_loss = 0.0
        print_step = 0
        print_mlm_loss = 0.0
        print_ita_loss = 0.0
        print_itm_loss = 0.0

        train_dataloader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_dataloader):
            model.train()
            # batch = {key: value.to(args.device) for key, value in batch.items()}
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            video_feature = batch["frame_input"].to(device, non_blocking=True)
            video_mask = batch['frame_mask'].to(device, non_blocking=True)

            if epoch > 1:
                alpha = config['alpha']
            else:
                alpha =config['alpha'] * min(1, i / len(train_dataloader))

            loss, (mlm_loss, itm_loss) = model(text_input_ids, text_mask, video_feature, video_mask, alpha)
            reduced_loss = reduce_tensor(loss.data)
            reduced_mlm_loss = reduce_tensor(mlm_loss.data)
            # reduced_ita_loss = reduce_tensor(ita_loss.data)
            reduced_itm_loss = reduce_tensor(itm_loss.data)

            loss = loss / config["accum_step"]
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), config["max_grad_norm"])
            #loss.backward()

            print_step += 1

            print_loss += reduced_loss.cpu().item()
            print_mlm_loss += reduced_mlm_loss.cpu().item()
            # print_ita_loss += reduced_ita_loss.cpu().item()
            print_itm_loss += reduced_itm_loss.cpu().item()
            if rank == 0 and print_step % config["print_steps"] == 0:
                lr = optimizer.param_groups[0]['lr']
                args.logger.info(f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : train total loss {print_loss/print_step:.5f}, "
                                 f"mlm_loss {print_mlm_loss/print_step:.5f}, ita_loss {print_ita_loss/print_step:.5f}, "
                                 f"itm_loss {print_itm_loss/print_step:.5f}.|| lr: {lr}.")

            if print_step % config["accum_step"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            args.logger.info(
            f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : train total loss {print_loss / print_step:.5f}, "
            f"mlm_loss {print_mlm_loss / print_step:.5f}, ita_loss {print_ita_loss / print_step:.5f}, "
            f"itm_loss {print_itm_loss / print_step:.5f}.|| lr: {lr} .")


            save_file = f'{args.pretrain_path}/pretrain_model_epoch_{epoch}.bin'
            model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            # torch.save(model_save, save_file)
            torch.save({"model_state_dict": model_save,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "amp_state_dict": amp.state_dict(),
                        "epoch": epoch}, save_file)
            args.logger.info(f"save mode to file {save_file}")


            args.logger.info(f"cost time: {time.time() - start_time}")

        dist.barrier()
    torch.cuda.empty_cache()


def main():

    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    args = parse_args()


    config = yaml.load(open("configs/Pretrain.yaml", 'r'), Loader=yaml.Loader)
    config["lr"] = float(config["lr"])
    config["other_lr"] = float(config["other_lr"])
    config["train_batch_size"] = config["train_batch_size"] // 2
    if rank == 0:
        print(f'lr={config["lr"]}, other_lr={config["other_lr"]}, train_batch_size={config["train_batch_size"]}')

    version = str(config["version"])
    log_path = os.path.join(args.log_path, "pretrain")
    filename = os.path.join(log_path, f"pretrain_{version}.log")


    args.logger = setup_logging(log_path, filename)
    args.pretrain_path = os.path.join(args.model_path, "pretrain", version)
    os.makedirs(args.pretrain_path, exist_ok=True)
    if rank == 0:
        args.logger.info("Training/evaluation parameters: %s, %s", args, config)
        yaml.dump(config, open(os.path.join(args.pretrain_path, 'config.yaml'), 'w'))


    train_worker(rank, local_rank, device, args, config)


if __name__ == '__main__':
    main()
