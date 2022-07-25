import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from utlils.category_id_map import category_id_to_lv2id


def create_dataloaders(args, train_index, val_index):
    # args.tf_idf_model = load_train_model(args.tf_idf_file)

    if train_index is not None and val_index is not None:

        train_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats, train_index)
        val_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats, val_index)
        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = RandomSampler(train_dataset, generator=torch.Generator())
        val_sampler = SequentialSampler(val_dataset)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=args.batch_size,
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = dataloader_class(val_dataset,
                                          batch_size=args.val_batch_size,
                                          sampler=val_sampler,
                                          drop_last=False)
    else:
        train_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats, None)

        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                       prefetch_factor=args.prefetch)
        else:
            # single-thread reading does not support prefetch_factor arg
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = RandomSampler(train_dataset, generator=torch.Generator())
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=args.batch_size,
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
                 zip_feats: str,
                 data_index: list = None,
                 test_mode: bool = False):


        self.max_frame = config["max_frames"]
        self.bert_seq_lenght = args.bert_seq_lenght
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers

        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        self.data_index = data_index

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns) if self.data_index is None else len(self.data_index)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
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
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask


    def __getitem__(self, idx: int) -> dict:

        # 转一下
        if self.data_index is not None:
            idx = self.data_index[idx]
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)
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


    def tokenize_text(self, text_dict):
        max_seq_length = self.bert_seq_lenght - 4  # cls + sep * 3
        each_max_seq_length = max_seq_length // 3

        title = text_dict["title"]
        asr = text_dict["asr"]
        ocr = ",".join([time_text["text"] for time_text in text_dict["ocr"]])

        title_sentences = self.tokenizer.tokenize(title)
        asr_sentences = self.tokenizer.tokenize(asr)
        ocr_sentences = self.tokenizer.tokenize(ocr)

        tokenized_sentence = ["[CLS]"] + title_sentences + ["[SEP]"] + asr_sentences + ["[SEP]"] + ocr_sentences
        if len(tokenized_sentence) > self.bert_seq_lenght - 1:
            tokenized_sentence = tokenized_sentence[:self.bert_seq_lenght-1]

        tokenized_sentence += ["[SEP]"] + ['[PAD]' for _ in range(self.bert_seq_lenght -1 - len(tokenized_sentence))]
        #
        #
        assert len(tokenized_sentence) == self.bert_seq_lenght
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_sentence), dtype=torch.long)
        attention_mask = torch.tensor([1 if tok != '[PAD]' else 0 for tok in tokenized_sentence], dtype=torch.long)

        return input_ids, attention_mask

