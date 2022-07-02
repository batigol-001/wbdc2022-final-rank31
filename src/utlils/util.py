import logging
from logging import handlers
import random
import os
import sys
import pickle
# import jieba
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(train_label):
    classes = np.unique(train_label)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_label)
    class_weights = dict(zip(classes, weights))
    return class_weights


from utlils.category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
# def setup_logging(args):

#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S',
#                         level=logging.INFO)
#     logger = logging.getLogger(__name__)
#

#     return logger


def setup_logging(log_path, filename):

    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger(filename)
    format_str = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(format_str)
    ch = handlers.RotatingFileHandler(filename=filename, mode='a', backupCount=10, encoding='utf-8')
    ch.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(ch)
    return logger


def build_optimizer(config, model):
    # 分层
    no_decay = ['bias', 'LayerNorm.weight']

    weight_decay = config['weight_decay']
    lr = config["lr"]
    other_lr = config["other_lr"]
    other_lr_layers = config["other_lr_layers"]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in other_lr_layers)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in other_lr_layers)], 'weight_decay': 0.0},

    ]

    # other lr
    classifier_params = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_layers)],
        'lr': other_lr,
        'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in other_lr_layers)],
            'lr': other_lr,
            'weight_decay': 0.0
        }
     ]

    optimizer_grouped_parameters.extend(classifier_params)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay,
                      eps=config["adam_epsilon"])
    return optimizer



def build_optimizer_v2(args, model, lr, other_lr):
    # 分层
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []

    multiplier = 1
    for layer in range(11, 0, -1):
        layer_params = [{
            'params': [p for n, p in model.named_parameters() if
                       f'encoder.layer.{layer}.' in n and not any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': args.weight_decay
        },
            {
                'params': [p for n, p in model.named_parameters() if
                           f'encoder.layer.{layer}.' in n and any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 0.0
            }
        ]

        optimizer_grouped_parameters.extend(layer_params)
        lr *= multiplier

    embedding_params = [{
        'params': [p for n, p in model.named_parameters() if
                   f'embeddings.' in n and not any(nd in n for nd in no_decay)],
        'lr': lr,
        'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       f'embeddings.' in n and any(nd in n for nd in no_decay)],
            'lr': lr,
            'weight_decay': 0.0
        }
     ]
    optimizer_grouped_parameters.extend(embedding_params)

    classifier_params = [{
        'params': [p for n, p in model.named_parameters() if 'linear' in n and not any(nd in n for nd in no_decay)],
        'lr': other_lr,
        'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if 'linear' in n and any(nd in n for nd in no_decay)],
            'lr': other_lr,
            'weight_decay': 0.0
        },
     ]

    optimizer_grouped_parameters.extend(classifier_params)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=args.weight_decay,
                      eps=args.adam_epsilon)

    if args.use_lookahead:
        optimizer = Lookahead(optimizer, 5, 0.5)

    # # frozen 前4层
    # for n, p in model.named_parameters():
    #     if 'encoder.layer.0.' in n or 'encoder.layer.1.' in n or 'encoder.layer.2.' in n or 'encoder.layer.3.' in n:
    #         p.requires_grad = False

    return optimizer



def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results


def save_model_class(model, path):
    with open(path, 'wb') as fp:
        f_str = pickle.dumps(model)
        fp.write(f_str)


def load_train_model(file):
    """
    :return 加载的模型:model_class_path
    """
    if os.path.isfile(file):
        with open(file, 'rb') as fp:
            model = pickle.loads(fp.read())
    else:
        logging.error("feat model not exists")
        sys.exit(1)

    return model

# def jieba_cut(x):
#     return " ".join(jieba.cut(x))


def get_w2v_embedding(model, sentence):
    vec = []
    sentence = sentence.split(" ")
    for w in sentence:
        if w in model.wv:
            vec.append(model.wv[w])
    if len(vec) > 0:
        return np.mean(vec, axis=0)
    else:
        return [0]*model.vector_size

class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM(object):
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]




class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        target = target.squeeze(dim=1)
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        target = target.squeeze(dim=1)
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = F.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

