import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from utlils.category_id_map import CATEGORY_ID_LIST
from third_party.masklm import MaskLM

from third_party.lxrt import LXRTXLayer, BertLayerNorm
from third_party.swin import swin_tiny


# 双流模型
class TwoStreamModel(nn.Module):
    def __init__(self, args, config):
        super(TwoStreamModel, self).__init__()
        bert_cfg = BertConfig.from_pretrained(args.bert_dir)

        bert_hidden_size = bert_cfg.hidden_size
        self.vocab_size = bert_cfg.vocab_size
        self.num_classes = len(CATEGORY_ID_LIST)

        embed_dim = config["embed_dim"]
        frame_embedding_size = config["frame_embedding_size"]
        self.momentum = config["momentum"]
        self.queue_size = config["queue_size"]
        temp = config["temp"]

        self.cross_layers_num = config["cross_layers_num"]

        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg, add_pooling_layer=True)
        #self.video_encoder = VisualFeatEncoder(bert_cfg, frame_embedding_size)
        self.video_encoder = swin_tiny(args.swin_pretrained_path)

        self.cross_layers = nn.ModuleList(
            [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
        )

        # 温度参数
        self.temp = nn.Parameter(torch.ones([]) * temp)

        self.vision_proj = nn.Linear(bert_hidden_size, embed_dim)
        self.text_proj = nn.Linear(bert_hidden_size, embed_dim)
        self.mlm_head = BertOnlyMLMHead(bert_cfg)

        self.itm_head = nn.Linear(bert_hidden_size, 2)

        self.lm = MaskLM(tokenizer_path=args.bert_dir)

        # 创建动量模型
        self.text_encoder_m = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_cfg, add_pooling_layer=True)
        self.video_encoder_m = swin_tiny(args.swin_pretrained_path)#self.video_encoder_m = VisualFeatEncoder(bert_cfg, frame_embedding_size)
        self.cross_layers_m = nn.ModuleList(
            [LXRTXLayer(bert_cfg) for _ in range(self.cross_layers_num)]
        )

        self.vision_proj_m = nn.Linear(bert_hidden_size, embed_dim)
        self.text_proj_m = nn.Linear(bert_hidden_size, embed_dim)
        self.mlm_head_m = BertOnlyMLMHead(bert_cfg)

        self.model_pairs = [[self.video_encoder, self.video_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_layers, self.cross_layers_m],
                            [self.mlm_head, self.mlm_head_m]
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("video_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.video_queue = nn.functional.normalize(self.video_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self,  text_input_ids, text_mask, video_feature, video_mask, alpha=0):

        # text_input_ids = inputs['text_input_ids']
        # text_mask = inputs['text_attention_mask']
        # video_feature = inputs["frame_input"]
        # video_mask = inputs['frame_mask']

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # 单模编码器, 输出video text embedding， [bs, 32, 768], [bs, 256, 768]
        video_embeds = self.video_encoder(video_feature)
        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        # feat 768-> 256 映射到低维空间， 视频取mean_pooling, 文本取[cls]
        video_feat = F.normalize(self.vision_proj(video_embeds.mean(1)), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # 动量编码器
        with torch.no_grad():
            self._momentum_update()
            video_embeds_m = self.video_encoder_m(video_feature)
            video_feat_m = F.normalize(self.vision_proj_m(video_embeds_m.mean(1)), dim=-1)
            # 合并队列
            video_feat_all = torch.cat([video_feat_m.t(), self.video_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m(input_ids=text_input_ids, attention_mask=text_mask)[
                "last_hidden_state"]
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # 计算相似度, 动量feat 与 正负样本， 负样本队列中的
            sim_i2t_m = video_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ video_feat_all / self.temp



            sim_targets = torch.zeros(sim_i2t_m.size()).to(video_feature.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets


        sim_i2t = video_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ video_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(video_feat_m, text_feat_m)

        with torch.no_grad():
            bs = video_feature.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            # 对角线正样本不要
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        video_embeds_neg = []
        video_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            video_embeds_neg.append(video_embeds[neg_idx])
            video_mask_neg.append(video_mask[neg_idx])
        video_embeds_neg = torch.stack(video_embeds_neg, dim=0)
        video_mask_neg = torch.stack(video_mask_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_mask_neg.append(text_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_mask_neg = torch.stack(text_mask_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_mask_all = torch.cat([text_mask, text_mask_neg], dim=0)

        video_embeds_all = torch.cat([video_embeds_neg, video_embeds], dim=0)
        video_mask_all = torch.cat([video_mask_neg, video_mask], dim=0)

        text_neg_outputs = text_embeds_all
        video_neg_outputs = video_embeds_all
        for layer_module in self.cross_layers:
            text_neg_outputs, video_neg_outputs = layer_module(text_neg_outputs, get_encoder_attention_mask(text_mask_all),
                                                               video_neg_outputs, get_encoder_attention_mask(video_mask_all))
        # 多模态融合, 参考LXMERT, cross_attention+self_attention+FFN
        text_outputs = text_embeds
        video_outputs = video_embeds
        for layer_module in self.cross_layers:
            text_outputs, video_outputs = layer_module(text_outputs, get_encoder_attention_mask(text_mask),
                                                       video_outputs, get_encoder_attention_mask(video_mask))
        # 取文本【CLS】
        vl_embeddings = torch.cat([text_outputs[:, 0, :], text_neg_outputs[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(video_feature.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        # MLM
        # MASK
        input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
        text_input_ids = input_ids.to(text_input_ids.device)
        lm_label = lm_label[:, 1:].to(text_input_ids.device)

        # MASK后再过一遍模型, 实现MLM
        text_embeds = self.text_encoder(input_ids=text_input_ids, attention_mask=text_mask)["last_hidden_state"]
        text_outputs = text_embeds
        video_outputs = video_embeds
        for layer_module in self.cross_layers:
            text_outputs, video_outputs = layer_module(text_outputs, get_encoder_attention_mask(text_mask),
                                                       video_outputs, get_encoder_attention_mask(video_mask))

        lm_prediction_scores = self.mlm_head(text_outputs)[:, 1:text_input_ids.size()[1], :]
        loss_mlm = nn.CrossEntropyLoss()(lm_prediction_scores.contiguous().view(-1, self.vocab_size),
                                         lm_label.contiguous().view(-1))

        # 动量
        with torch.no_grad():
            text_outputs_m = text_embeds
            video_outputs_m = video_embeds_m
            for layer_module in self.cross_layers_m:
                text_outputs_m, video_outputs_m = layer_module(text_outputs_m, get_encoder_attention_mask(text_mask),
                                                               video_outputs_m, get_encoder_attention_mask(video_mask))
            lm_prediction_scores_m = self.mlm_head_m(text_outputs_m)[:, 1:text_input_ids.size()[1], :]

        soft_labels = F.softmax(lm_prediction_scores_m, dim=-1)
        loss_mlm_distill = -torch.sum(F.log_softmax(lm_prediction_scores, dim=-1) * soft_labels, dim=-1)
        loss_mlm_distill = loss_mlm_distill[lm_label != -100].mean()

        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_mlm_distill

        loss = (loss_mlm + loss_ita + loss_itm * 10)/3
        return loss, (loss_mlm, loss_ita, loss_itm)


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
    def _dequeue_and_enqueue(self, video_feat, text_feat):
        # gather keys before updating queue
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)

        # video_feats = video_feat
        # text_feats = text_feat
        batch_size = video_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.video_queue[:, ptr:ptr + batch_size] = video_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr




class VisualFeatEncoder(nn.Module):
    def __init__(self, config, feat_dim):
        super().__init__()

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def forward(self, visn_input):
        x = self.visn_fc(visn_input)
        x = self.visn_layer_norm(x)
        output = self.dropout(x)
        return output


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def get_encoder_attention_mask(mask):
    encoder_mask = mask[:, None, None, :]
    encoder_mask = (1.0 - encoder_mask) * -10000.0

    return encoder_mask

