# %%writefile data/masklm.py
import math
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer


class MaskLM(object):
    def __init__(self, tokenizer_path='bert-base-chinese', mlm_probability=0.15):
        self.mlm_probability = 0.15
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def  torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class MaskVideo(object):
    def __init__(self, mlm_probability=0.15):
        self.mlm_probability = 0.15

    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask

        masked_indices = torch.bernoulli(probability_matrix).bool()

        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2))

        return inputs, video_labels_index


class ShuffleVideo(object):
    def __init__(self):
        pass

    def torch_shuf_video(self, video_feature, video_mask):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        video_mask = video_mask[shuf_index]

        return video_feature, video_mask, label




def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.hidden_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

