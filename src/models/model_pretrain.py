import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from utlils.category_id_map import CATEGORY_ID_LIST
from third_party.masklm import MaskLM, MaskVideo, VisualOnlyMLMHead, ShuffleVideo

from third_party.lxrt import LXRTXLayer, BertLayerNorm
from third_party.swin import swin


# 双流模型
class TwoStreamModel(nn.Module):
    def __init__(self, args, config):
        super(TwoStreamModel, self).__init__()
        bert_cfg = BertConfig.from_pretrained(args.bert_dir)

        bert_hidden_size = bert_cfg.hidden_size
        self.vocab_size = bert_cfg.vocab_size
        self.num_classes = len(CATEGORY_ID_LIST)

        frame_embedding_size = config["frame_embedding_size"]

        self.cross_layers_num = config["cross_layers_num"]

        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg, add_pooling_layer=True)

        self.video_encoder = swin(args.swin_pretrained_path)
        self.video_proj_linear = nn.Linear(frame_embedding_size, bert_hidden_size)

        self.cross_layers = nn.ModuleList(
            [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
        )

        # mlm
        self.lm = MaskLM(tokenizer_path=args.bert_dir)
        self.mlm_head = BertOnlyMLMHead(bert_cfg)


        # itm
        self.sv = ShuffleVideo()
        self.itm_head = nn.Linear(bert_hidden_size, 1)



    def forward(self,  text_input_ids, text_mask, video_feature, video_mask, alpha=0):


        # MLM - MASK
        input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
        text_input_ids = input_ids.to(text_input_ids.device)
        lm_label = lm_label[:, 1:].to(text_input_ids.device)

        # 先item shuffle, 回传出来shuffle后的video_feature, video_mask
        input_feature, input_mask, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu(),
                                                                                     video_mask.cpu())
        video_feature = input_feature.to(video_feature.device)
        video_mask = input_mask.to(video_mask.device)
        video_text_match_label = video_text_match_label.to(video_feature.device)


        # 模型
        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        # video_embeds = self.video_encoder(video_feature)
        # video_embeds = self.video_proj_linear(video_embeds)
        video_embeds = nn.Tanh()(self.video_proj_linear(video_feature))
        # 多模态融合, 参考LXMERT, cross_attention+self_attention+FFN
        text_outputs = text_embeds
        video_outputs = video_embeds
        for layer_module in self.cross_layers:
            text_outputs, video_outputs = layer_module(text_outputs, get_encoder_attention_mask(text_mask),
                                                       video_outputs, get_encoder_attention_mask(video_mask))

        encoder_outputs = torch.cat((text_outputs, video_outputs), dim=1)
        # mlm-loss
        lm_prediction_scores = self.mlm_head(text_embeds + text_outputs)[:, 1:, :]
        loss_mlm = nn.CrossEntropyLoss()(lm_prediction_scores.contiguous().view(-1, self.vocab_size),
                                         lm_label.contiguous().view(-1))


        # itm-loss
        # pred = self.itm_head(text_outputs[:, 0, :])
        pred = self.itm_head(encoder_outputs.mean(1))
        itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))

        loss = (loss_mlm + itm_loss * 10)/2
        return loss, (loss_mlm, itm_loss)




@torch.no_grad()
def get_encoder_attention_mask(mask):
    encoder_mask = mask[:, None, None, :]
    encoder_mask = (1.0 - encoder_mask) * -10000.0

    return encoder_mask

