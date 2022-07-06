
import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPooler
from utlils.category_id_map import CATEGORY_ID_LIST
from third_party.masklm import MaskLM, MaskVideo, ShuffleVideo

from third_party.lxrt import LXRTXLayer, BertLayer, BertLayerNorm
from third_party.swin import swin_tiny
from third_party.vit import vit

# 双流模型
class TwoStreamModel(nn.Module):
    def __init__(self, args, config):
        super(TwoStreamModel, self).__init__()
        bert_cfg = BertConfig.from_pretrained(args.bert_dir)

        bert_hidden_size = bert_cfg.hidden_size
        self.vocab_size = bert_cfg.vocab_size
        self.num_classes = len(CATEGORY_ID_LIST)
        frame_embedding_size = config["frame_embedding_size"]

        self.momentum = config["momentum"]
        self.queue_size = config["queue_size"]
        temp = config["temp"]

        self.cross_layers_num = config["cross_layers_num"]

        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg)
        # self.video_encoder = VisualFeatEncoder(bert_cfg, frame_embedding_size)

        self.video_encoder = swin_tiny(args.swin_pretrained_path)




        self.cross_layers = nn.ModuleList(
            [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
        )

        # 温度参数
        # self.temp = nn.Parameter(torch.ones([]) * temp)

        #  分类头
        self.cls_linear = torch.nn.Linear(bert_hidden_size * 2, self.num_classes)

        self.is_distrill = config["distrill"]

        if self.is_distrill:
            # 创建动量模型

            self.text_encoder_m = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg)
            self.video_encoder_m = swin_tiny(args.swin_pretrained_path)# VisualFeatEncoder(bert_cfg, frame_embedding_size)
            self.cross_layers_m = nn.ModuleList(
                [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
            )

            self.cls_linear_m = torch.nn.Linear(bert_hidden_size * 2, self.num_classes)

            self.model_pairs = [[self.video_encoder, self.video_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.cross_layers, self.cross_layers_m],
                                [self.cls_linear, self.cls_linear_m]
                                ]
            self.copy_params()

    def forward(self, text_input_ids, text_mask, video_feature, video_mask, labels=None, alpha=0):

        # text_input_ids = inputs['text_input_ids']
        # text_mask = inputs['text_attention_mask']
        # video_feature = inputs["frame_input"]
        # video_mask = inputs['frame_mask']

        # 单模编码器, 输出video text embedding， [bs, 32, 768], [bs, 256, 768]
        video_embeds = self.video_encoder(video_feature)
        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        # 多模态融合, 参考LXMERT, cross_attention+self_attention+FFN
        text_outputs = text_embeds
        video_outputs = video_embeds
        for layer_module in self.cross_layers:
            text_outputs, video_outputs = layer_module(text_outputs, get_encoder_attention_mask(text_mask),
                                                       video_outputs, get_encoder_attention_mask(video_mask))
        #
        text_feats = text_outputs.mean(1)
        video_feats = video_outputs.mean(1)
        concatenate_pooling = torch.cat((text_feats, video_feats), dim=-1)
        concatenate_pooling = nn.Dropout(0.2)(concatenate_pooling)
        # concatenate_pooling = nn.Dropout(0.2)(text_outputs[:, 0, :])
        preds = self.cls_linear(concatenate_pooling)

        if labels is None:
            return F.log_softmax(preds, dim=-1)
        else:
            labels = labels.squeeze(dim=1)
            loss = F.cross_entropy(preds, labels)

            if self.is_distrill:
                # 动量编码器
                with torch.no_grad():
                    self._momentum_update()
                    video_embeds_m = self.video_encoder_m(video_feature)
                    # text_embeds_m = self.text_encoder_m(input_ids=text_input_ids, attention_mask=text_mask)[
                    #     "last_hidden_state"]

                    text_outputs_m = text_embeds
                    video_outputs_m = video_embeds_m

                    for layer_module in self.cross_layers_m:
                        text_outputs_m, video_outputs_m = layer_module(text_outputs_m,
                                                                   get_encoder_attention_mask(text_mask),
                                                                   video_outputs_m,
                                                                   get_encoder_attention_mask(video_mask))

                    text_feats_m = text_outputs_m.mean(1)
                    video_feats_m = video_outputs_m.mean(1)
                    concatenate_pooling_m = torch.cat((text_feats_m, video_feats_m), dim=-1)
                    concatenate_pooling_m = nn.Dropout(0.2)(concatenate_pooling_m)
                    # concatenate_pooling_m = nn.Dropout(0.2)(text_outputs_m[:, 0, :])
                    preds_m = self.cls_linear_m(concatenate_pooling_m)

                soft_labels = F.softmax(preds_m, dim=-1)
                loss_distill = -torch.sum(F.log_softmax(preds, dim=-1) * soft_labels, dim=-1).mean()

                loss = (1 - alpha) * loss + alpha * loss_distill

            with torch.no_grad():
                pred_label_id = torch.argmax(preds, dim=1)
                accuracy = (labels == pred_label_id).float().sum() / labels.shape[0]

            return loss, accuracy, pred_label_id, labels




    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


@torch.no_grad()
def get_encoder_attention_mask(mask):
    encoder_mask = mask[:, None, None, :]
    encoder_mask = (1.0 - encoder_mask) * -10000.0

    return encoder_mask


