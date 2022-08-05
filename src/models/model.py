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
from third_party.swin import swin
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

        self.cross_layers_num = config["cross_layers_num"]

        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg,
                                                      add_pooling_layer=False)

        self.video_encoder = swin(args.swin_pretrained_path)
        self.video_proj_linear = nn.Linear(frame_embedding_size, bert_hidden_size)

        self.cross_layers = nn.ModuleList(
            [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
        )

        #  分类头
        self.cls_linear = nn.Linear(bert_hidden_size * 2, self.num_classes)

    def forward(self, text_input_ids, text_mask, video_feature, video_mask, labels=None, alpha=0):

        # 单模编码器, 输出video text embedding， [bs, 32, 768], [bs, 256, 768]

        video_embeds = self.video_encoder(video_feature)
        video_embeds = self.video_proj_linear(video_embeds)

        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        # 多模态融合, 参考LXMERT, cross_attention+self_attention+FFN
        text_embeds_mean = text_embeds.mean(1)  # (text_embeds * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1).unsqueeze(-1)
        video_embeds_mean = video_embeds.mean(1)

        text_outputs = text_embeds
        video_outputs = video_embeds
        for layer_module in self.cross_layers:
            text_outputs, video_outputs = layer_module(text_outputs, get_encoder_attention_mask(text_mask),
                                                       video_outputs, get_encoder_attention_mask(video_mask))

        text_outputs_mean = text_outputs.mean(1)
        video_outputs_mean = video_outputs.mean(1)

        text_mean_feats = text_outputs_mean + text_embeds_mean
        video_mean_feats = video_outputs_mean + video_embeds_mean

        concat_feats = torch.cat((text_mean_feats, video_mean_feats), dim=-1)

        preds = self.cls_linear(concat_feats)

        if labels is None:
            return preds  # F.log_softmax(preds, dim=-1)
        else:
            labels = labels.squeeze(dim=1)
            loss = F.cross_entropy(preds, labels)

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
    encoder_mask = encoder_mask.half()
    return encoder_mask

