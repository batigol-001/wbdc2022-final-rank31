
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


        self.fusion_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg).encoder

        #  分类头
        self.cls_linear = nn.Linear(bert_hidden_size * 4, self.num_classes)





    def forward(self, text_input_ids, text_mask, video_feature, video_mask, labels=None, alpha=0):

        # 单模编码器, 输出video text embedding， [bs, 32, 768], [bs, 256, 768]

        video_embeds = self.video_encoder(video_feature)
        video_embeds = self.video_proj_linear(video_embeds)

        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]


        fusion_embeds = torch.cat((text_embeds, video_embeds), dim=1)
        fusion_mask = get_encoder_attention_mask(torch.cat((text_mask, video_mask), dim=1))
        fusion_outputs = self.fusion_encoder(hidden_states=fusion_embeds, attention_mask=fusion_mask,  output_hidden_states=True)

        # 后4层concat
        all_hidden_states = fusion_outputs['hidden_states']
        concat_feats = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )  # [bs, seq_len, hidden_dim * 4]
        concat_feats = concat_feats.mean(1)
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


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask=None):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x

