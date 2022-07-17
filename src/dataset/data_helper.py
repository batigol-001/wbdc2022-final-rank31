import os
import json
import random
import zipfile
from io import BytesIO
from functools import partial
from PIL import Image


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor, RandomResizedCrop, RandomCrop, RandomHorizontalFlip

from utlils.category_id_map import category_id_to_lv2id
from dataset.randzaugment import RandomAugment

import cv2
class GaussianBlur(object):
    def __init__(self, K_size=3, sigmar=3, isPIL=True):
        self.isPIL = isPIL
        self.K_size = K_size
        self.sigmar = sigmar

    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)
        img = cv2.GaussianBlur(img, (self.K_size, self.K_size), self.sigmar)
        return img

def create_dataloaders(args, config, train_index, val_index):
    # args.tf_idf_model = load_train_model(args.tf_idf_file)

    if train_index is not None and val_index is not None:

        train_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, train_index)
        val_dataset = MultiModalDataset(args, config, args.train_annotation, args.train_zip_frames, val_index)
        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = RandomSampler(train_dataset, generator=torch.Generator())
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


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 config,
                 ann_path: str,
                 zip_frame_dir: str,
                 data_index: list = None,
                 test_mode: bool = False):


        self.bert_seq_length = args.bert_seq_length

        self.max_frame = config["max_frames"]
        self.test_mode = test_mode

        self.zip_frame_dir = zip_frame_dir
        self.num_workers = args.num_workers


        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        self.data_index = data_index

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        # we use the standard image transform as in the offifical Swin-Transformer.
        # if is_augment:
        #     self.transform = Compose([
        #         Resize(256),
        #         RandomCrop(224),
        #         RandomHorizontalFlip(p=0.3),
        #         GaussianBlur(),
        #         RandomAugment(2, 9, isPIL=False, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
        #                                               'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        #         ToTensor(),
        #         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])
        # else:
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            # GaussianBlur(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])




    def __len__(self) -> int:
        return len(self.anns) if self.data_index is None else len(self.data_index)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def __getitem__(self, idx: int) -> dict:

        # 转一下
        if self.data_index is not None:
            idx = self.data_index[idx]

        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)
        # Step 2, load title tokens
        text_input_ids, text_attention_mask = self.tokenize_text(self.anns[idx])



        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

            # data['label_l1'] = torch.LongTensor([lv2id_to_lv1id(label)])


        return data

    def get_other_feats(self, text_dict):
        title = text_dict["title"]
        asr = text_dict["asr"]
        ocr = ",".join([time_text["text"] for time_text in text_dict["ocr"]])


        # 长尾分布
        title_len = np.log1p(len(title)) / 7.1
        asr_len = np.log1p(len(asr)) / 7.3
        ocr_len = np.log1p(len(ocr)) / 9.6
        ocr_unique = (len(text_dict["ocr"]) + 1) / 32
        ocr_time = [time_text["time"] for time_text in text_dict["ocr"]]
        ocr_max_time = (max(ocr_time) + 1 if ocr_time else 0)/ 32
        other_feats = [title_len, asr_len, ocr_len, ocr_unique, ocr_max_time]
        other_feats = torch.tensor(other_feats, dtype=torch.float)

        return other_feats




    def tokenize_text(self, text_dict):
        max_seq_length = self.bert_seq_length - 4  # cls + sep * 3
        each_max_seq_length = max_seq_length // 3

        title = text_dict["title"]
        asr = text_dict["asr"]
        ocr = ",".join([time_text["text"] for time_text in text_dict["ocr"]])

        title_sentences = self.tokenizer.tokenize(title)
        asr_sentences = self.tokenizer.tokenize(asr)
        ocr_sentences = self.tokenizer.tokenize(ocr)

        # num_title = len(title_sentences)
        # num_asr = len(asr_sentences)
        # num_ocr = len(ocr_sentences)
        # total_num_sent = num_title + num_asr + num_ocr
        # if total_num_sent > max_seq_length:
        #     print("trucncate_mode:", self.bert_truncate_mode)
        #     if self.bert_truncate_mode == 0:
        #         title_sentences = title_sentences[:each_max_seq_length]
        #         asr_sentences = asr_sentences[:each_max_seq_length]
        #         ocr_sentences = ocr_sentences[:each_max_seq_length]
        #     else:
        #         _truncate_seq_pair(title_sentences, asr_sentences, ocr_sentences, max_seq_length)
        #
        # final_total_sent = len(title_sentences) + len(asr_sentences) + len(ocr_sentences)
        # # 不够的补pad, 截断可能多截断了一点,也补pad

        #
        # tokenized_sentence = ["[CLS]"] + title_sentences + ["[SEP]"] + asr_sentences + ["[SEP]"] + ocr_sentences + ["[SEP]"] \
        #                      + ['[PAD]' for _ in range(max_seq_length - final_total_sent)]

        tokenized_sentence = ["[CLS]"] + title_sentences + ["[SEP]"] + asr_sentences + ["[SEP]"] + ocr_sentences
        if len(tokenized_sentence) > self.bert_seq_length - 1:
            tokenized_sentence = tokenized_sentence[:self.bert_seq_length-1]

        tokenized_sentence += ["[SEP]"] + ['[PAD]' for _ in range(self.bert_seq_length -1 - len(tokenized_sentence))]


        assert len(tokenized_sentence) == self.bert_seq_length
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_sentence), dtype=torch.long)
        attention_mask = torch.tensor([1 if tok != '[PAD]' else 0 for tok in tokenized_sentence], dtype=torch.long)

        return input_ids, attention_mask


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_c) >= max(len(tokens_a), len(tokens_b)):
            tokens_c.pop()

        elif len(tokens_b) >= max(len(tokens_a), len(tokens_c)):
            tokens_b.pop()
        else:
            tokens_a.pop()
