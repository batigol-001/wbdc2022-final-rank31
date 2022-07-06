import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
sys.path.append(project_path)

# import ruamel_yaml as yaml
import yaml
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from functools import partial
from configs.config import parse_args
from utlils.util import setup_device, setup_seed, build_optimizer, setup_logging
from utlils.util import  init_distributed_mode, get_rank
from dataset.data_helper import MultiModalDataset
from models.model_pretrain import TwoStreamModel

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


def create_pretrain_dataloaders(args,  config, annotations, zip_frames, val_size=None):
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    dataset = None
    for idx, (_annotation, _zip_feat) in enumerate(zip(annotations, zip_frames)):

        # index = list(range(5000))
        sub_dataset = MultiModalDataset(args, config, _annotation, _zip_feat, None, test_mode=True)
        if idx == 0:
            dataset = sub_dataset
        else:
            dataset = ConcatDataset([dataset, sub_dataset])
    val_dataloader = None
    if val_size is not None and val_size > 0:
        size = len(dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                                   generator=torch.Generator().manual_seed(args.seed))

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=config["train_batch_size"],
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = dataloader_class(val_dataset,
                                          batch_size=config["val_batch_size"],
                                          sampler=val_sampler,
                                          drop_last=False)
    else:
        train_sampler = RandomSampler(dataset)
        train_dataloader = dataloader_class(dataset,
                                        batch_size=config["train_batch_size"],
                                        sampler=train_sampler,
                                        drop_last=True)

    return train_dataloader, val_dataloader


def train(args, config):
    #  load data
    # annotations = [args.unlabeled_annotation, args.train_annotation]
    # zip_feats = [args.unlabeled_zip_frames, args.train_zip_frames]

    annotations = [args.train_annotation]
    zip_feats = [args.train_zip_frames]

    train_dataloader, val_dataloader = create_pretrain_dataloaders(args, config, annotations, zip_feats, val_size=None)


    epochs = config["max_epochs"]
    setps_per_epoch = len(train_dataloader)
    max_steps = epochs * setps_per_epoch
    num_warmup_steps = int(max_steps * config["warmup_ratio"])

    args.logger.info(f"total epochs: {epochs}")
    args.logger.info(f"total steps: {max_steps}")
    args.logger.info(f"steps per epochs: {setps_per_epoch}")
    args.logger.info(f"warmup steps: {num_warmup_steps}")

    model = TwoStreamModel(args, config)
    # for n, m in model.named_parameters():
    #     print(n)

    # exist_pretrain_file = "/root/autodl-nas/pretrain/1.0/pretrain_model_epoch_10.bin"
    # if os.path.exists(exist_pretrain_file):
    #     args.logger.info(f"加载已经预训练过的模型, file= {exist_pretrain_file}")
    #     model.load_state_dict(torch.load(exist_pretrain_file), strict=False)


    optimizer = build_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps)
    model = model.to(args.device)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()

    if args.n_gpu > 1:
        model = torch.nn.parallel.DataParallel(model)


    best_loss = 1000
    for epoch in range(1, epochs+1):
        start_time = time.time()
        print_loss = 0.0
        print_step = 0
        print_mlm_loss = 0.0
        print_ita_loss = 0.0
        print_itm_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            model.train()
            # batch = {key: value.to(args.device) for key, value in batch.items()}
            text_input_ids = batch['text_input_ids'].to(args.device)
            text_mask = batch['text_attention_mask'].to(args.device)
            video_feature = batch["frame_input"].to(args.device)
            video_mask = batch['frame_mask'].to(args.device)


            if epoch > 1:
                alpha = config['alpha']
            else:
                alpha =config['alpha'] * min(1, i / len(train_dataloader))

            loss, (mlm_loss,  ita_loss, itm_loss) = model(text_input_ids, text_mask, video_feature, video_mask, alpha)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print_step += 1

            print_loss += loss.item()
            print_mlm_loss += mlm_loss.mean().item()
            print_ita_loss += ita_loss.mean().item()
            print_itm_loss += itm_loss.mean().item()
            if print_step % config["print_steps"] == 0:
                args.logger.info(f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : train total loss {print_loss/print_step:.5f}, "
                                 f"mlm_loss {print_mlm_loss/print_step:.5f}, ita_loss {print_ita_loss/print_step:.5f}, "
                                 f"itm_loss {print_itm_loss/print_step:.5f}.|| best_loss: {best_loss}.")

        args.logger.info(
            f"Epoch {epoch} step [{print_step} / {setps_per_epoch}] : train total loss {print_loss / print_step:.5f}, "
            f"mlm_loss {print_mlm_loss / print_step:.5f}, ita_loss {print_ita_loss / print_step:.5f}, "
            f"itm_loss {print_itm_loss / print_step:.5f}.|| best_loss: {best_loss} .")


        save_file = f'{args.pretrain_path}/pretrain_model_epoch_{epoch}.bin'
        model_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_save, save_file)
        args.logger.info(f"save mode to file {save_file}")


        if val_dataloader is not None:
            args.logger.info("开始评估！")
            loss, (mlm_loss, ita_loss, itm_loss) = validate(model, val_dataloader)

            args.logger.info(
                f"Validate  : total loss {loss :.5f}, "
                f"mlm_loss {mlm_loss :.5f}, ita_loss {ita_loss :.5f}, "
                f"itm_loss {itm_loss :.5f}.")
            if loss < best_loss:
                best_loss = loss
        # todo
        gc.collect()
        args.logger.info(f"cost time: {time.time() - start_time}")

    torch.cuda.empty_cache()


def main():

    args = parse_args()

    config = yaml.load(open("configs/Pretrain.yaml", 'r'), Loader=yaml.Loader)
    config["lr"] = float(config["lr"])
    config["other_lr"] = float(config["other_lr"])
    
    version = str(config["version"])
    log_path = os.path.join(args.log_path, "pretrain")
    filename = os.path.join(log_path, f"pretrain_{version}.log")


    args.logger = setup_logging(log_path, filename)

    setup_device(args)
    setup_seed(args)
    args.pretrain_path = os.path.join(args.model_path, "pretrain", version)
    os.makedirs(args.pretrain_path, exist_ok=True)

    args.logger.info("Training/evaluation parameters: %s, %s", args, config)

    yaml.dump(config, open(os.path.join(args.pretrain_path, 'config.yaml'), 'w'))


    train(args, config)


if __name__ == '__main__':
    main()
