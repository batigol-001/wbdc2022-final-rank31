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
from third_party.swin import  swin
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

        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg, add_pooling_layer=False)

        self.video_encoder = swin(args.swin_pretrained_path)
        self.video_proj_linear = nn.Sequential(nn.Linear(frame_embedding_size, bert_hidden_size))

        # bert，随机初始化
        fusion_config = BertConfig()
        fusion_config.num_hidden_layers = 3
        fusion_config.intermediate_size = 3072
        fusion_config.hidden_size = 768
        fusion_config.num_attention_heads = 12
        fusion_config.vocab_size = 5
        self.cross_layers = BertModel(config=fusion_config)
        self.cross_layers.apply(self._init_weights)


        #  分类头
        self.cls_linear = nn.Linear(bert_hidden_size * 3, self.num_classes)

        self.is_distrill = config["distrill"]

        if self.is_distrill:
            # 创建动量模型

            self.text_encoder_m = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg, add_pooling_layer=False)
            self.video_encoder_m = swin(args.swin_pretrained_path)
            self.video_proj_linear_m = nn.Sequential(nn.Linear(frame_embedding_size, bert_hidden_size))

            self.cross_layers_m = nn.ModuleList(
                [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
            )

            self.cls_linear_m = nn.Linear(bert_hidden_size * 2, self.num_classes)

            self.model_pairs = [[self.video_encoder, self.video_encoder_m],
                                [self.video_proj_linear, self.video_proj_linear_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.cross_layers, self.cross_layers_m],
                                [self.cls_linear, self.cls_linear_m]
                                ]
            self.copy_params()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)



    def forward(self, text_input_ids, text_mask, video_feature, video_mask, labels=None, alpha=0):


        # 单模编码器, 输出video text embedding， [bs, 32, 768], [bs, 256, 768]
        video_embeds = self.video_encoder(video_feature)
        video_embeds = self.video_proj_linear(video_embeds)

        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        text_embeds_mean = text_embeds.mean(1)#(text_embeds * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1).unsqueeze(-1)
        video_embeds_mean = video_embeds.mean(1)#(video_embeds * video_mask.unsqueeze(-1)).sum(1) / video_mask.sum(1).unsqueeze(-1)

        # 多模态融合, 参考LXMERT, cross_attention+self_attention+FFN
        fusion_embeds = torch.cat((text_embeds, video_embeds), dim=1)
        fusion_mask = torch.cat((text_mask, video_mask), dim=1)
        fusion_outputs = self.cross_layers(inputs_embeds=fusion_embeds, attention_mask=fusion_mask)["last_hidden_state"]
        fusion_outputs_mean = fusion_outputs.mean(1)#(fusion_outputs * fusion_mask.unsqueeze(-1)).sum(1) / fusion_mask.sum(1).unsqueeze(-1)


        concat_feats = torch.cat((text_embeds_mean, video_embeds_mean, fusion_outputs_mean), dim=-1)

        preds = self.cls_linear(concat_feats)


        if labels is None:
            return preds #F.log_softmax(preds, dim=-1)
        else:
            labels = labels.squeeze(dim=1)
            loss = F.cross_entropy(preds, labels)

            if self.is_distrill:
                # 动量编码器
                with torch.no_grad():
                    self._momentum_update()
                    video_embeds_m = self.video_encoder_m(video_feature)
                    video_embeds_m = self.video_proj_linear_m(video_embeds_m)
                    # text_embeds_m = self.text_encoder_m(input_ids=text_input_ids, attention_mask=text_mask)[
                    #     "last_hidden_state"]

                    text_outputs_m = text_embeds
                    video_outputs_m = video_embeds_m

                    for layer_module in self.cross_layers_m:
                        text_outputs_m, video_outputs_m = layer_module(text_outputs_m,
                                                                   get_encoder_attention_mask(text_mask),
                                                                   video_outputs_m,
                                                                   get_encoder_attention_mask(video_mask))

                    text_feats_m = text_outputs_m.mean(1) + text_embeds.mean(1)
                    video_feats_m = video_outputs_m.mean(1) + video_embeds_m.mean(1)
                    concat_feats_m = torch.cat((text_feats_m, video_feats_m), dim=-1)

                    preds_m = self.cls_linear_m(concat_feats_m)

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


class VideoGRU(torch.nn.Module):
    def __init__(self, video_frame_size, dropout=0.2):
        super(VideoGRU, self).__init__()
        self.embedding_size = video_frame_size
        self.dropout = torch.nn.Dropout(dropout)
        self.GRU_layer = torch.nn.GRU(video_frame_size, video_frame_size, batch_first=True, bidirectional=True)
        self.linear_layer = torch.nn.Linear(video_frame_size * 2, video_frame_size)
        self.layer_norm = nn.LayerNorm(video_frame_size, eps=1e-12)

    def forward(self, video_embeds, video_mask):

        embedded_vector = self.dropout(video_embeds)
        video_length = video_mask.sum(dim=-1).detach().cpu().numpy().astype("int64")
        # torch.nn.utils.rnn.pack_padded_sequence()这里的pack，理解成压紧比较好。 将一个 填充过的变长序列 压紧。（填充时候，会有冗余，所以压紧一下）其中pack的过程为：（注意pack的形式，不是按行压，而是按列压）
        packed_embedded_vector = torch.nn.utils.rnn.pack_padded_sequence(embedded_vector,
                                                                         video_length,
                                                                         batch_first=True,
                                                                         enforce_sorted=False)
        # packed_out的vectorsize从【batch_size，pad_x_len，embedding_size】变为【batch_size，pad_x_len，en_hid_size*2】
        packed_out, _ = self.GRU_layer(packed_embedded_vector)
        # torch.nn.utils.rnn.pad_packed_sequence()填充packed_sequence。上面提到的函数的功能是将一个填充后的变长序列压紧。 这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。填充时会初始化为0。
        # out_vector的vectorsize没变，与packed_out的相同
        out_vector, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out_vector = torch.tanh(self.linear_layer(out_vector))

        out_vector = self.layer_norm(out_vector + video_embeds)


        return out_vector
