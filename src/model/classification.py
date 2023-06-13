from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rex.modules.cnn import MultiKernelCNN
from rex.modules.ffn import FFN
from transformers import BertModel, BertTokenizerFast
from src.model.loss import ResampleLoss


# CLS
class BertCLS(nn.Module):
    def __init__(
        self,
        device: int,
        plm_filepath: str,
        num_classes: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        mid_dims=[
            100,
        ],
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        
        self.ffn = FFN(
            input_dim=hidden_size,
            output_dim=num_classes,
            mid_dims=mid_dims,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        ) 
        self.loss = nn.BCELoss(reduction=reduction)
        self.partial_loss = nn.BCELoss(reduction='none')
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.ffn(cls_hidden)

        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs.sigmoid(), labels.float())
            elif train_type == 'pst_train':
                # partial loss
                pseudo_label = labels.masked_fill(labels == 2, 1.0)
                pseudo_loss = self.partial_loss(outs.sigmoid(), pseudo_label.float())
                masked_loss = pseudo_loss.masked_fill(labels == 2, 0.0)
                results["loss"] = masked_loss.mean()
            else:
                raise ValueError(f'not define train_type {train_type}')

        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.ffn(cls_hidden)

        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        return results


# TextCNN
class TextCNN(nn.Module):
    def __init__(
        self,
        device: int,
        plm_filepath: str,
        num_filters: Optional[int] = 300,
        num_classes: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        kernel_sizes=[1, 3, 5],
        mid_dims=[
            100,
        ],
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        
        self.cnn = MultiKernelCNN(
            in_channel=hidden_size,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.ffn = FFN(
            input_dim=num_filters * len(kernel_sizes),
            output_dim=num_classes,
            mid_dims=mid_dims,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )
        self.loss = nn.BCELoss(reduction=reduction)
        self.partial_loss = nn.BCELoss(reduction='none')
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs.sigmoid(), labels.float())
            elif train_type == 'pst_train':
                # partial loss
                pseudo_label = labels.masked_fill(labels == 2, 1.0)
                pseudo_loss = self.partial_loss(outs.sigmoid(), pseudo_label.float())
                masked_loss = pseudo_loss.masked_fill(labels == 2, 0.0)
                results["loss"] = masked_loss.mean()
            else:
                raise ValueError(f'not define train_type {train_type}')
        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        return results
   

# CLS+各种LOSS
class SetLoss(nn.Module):
    def __init__(
        self,
        device: int,
        class_freq: list,
        train_num: int,
        plm_filepath: str,
        loss_func_name: str,
        num_classes: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        mid_dims=[
            100,
        ],
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        
        self.ffn = FFN(
            input_dim=hidden_size,
            output_dim=num_classes,
            mid_dims=mid_dims,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )
        
        if loss_func_name == 'FL':  # FL
            self.loss = ResampleLoss(device=self.device, reweight_func=None, loss_weight=1.0,partial=False,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(),
                                    class_freq=class_freq, train_num=train_num) 
            self.partial_loss = ResampleLoss(device=self.device, reweight_func=None, loss_weight=1.0,partial=True,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(),
                                    class_freq=class_freq, train_num=train_num) 
        if loss_func_name == 'RFL': # R-FL
            self.loss = ResampleLoss(device=self.device, reweight_func='rebalance', loss_weight=1.0,partial=False,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=class_freq, train_num=train_num)
            self.partial_loss = ResampleLoss(device=self.device, reweight_func='rebalance', loss_weight=1.0,partial=True,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),
                                    logit_reg=dict(),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=class_freq, train_num=train_num)
        if loss_func_name == 'CB': # CB
            self.loss = ResampleLoss(device=self.device, reweight_func='CB', loss_weight=5.0,reduction=reduction,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),partial=False,
                                    logit_reg=dict(),
                                    CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                    class_freq=class_freq, train_num=train_num) 
            self.partial_loss = ResampleLoss(device=self.device, reweight_func='CB', loss_weight=5.0,reduction=reduction,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),partial=True,
                                    logit_reg=dict(),
                                    CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                    class_freq=class_freq, train_num=train_num) 
        if loss_func_name == 'DB': # DB
            self.loss = ResampleLoss(device=self.device, reweight_func='rebalance', loss_weight=1.0,reduction=reduction,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),partial=False,
                                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=class_freq, train_num=train_num)
            self.partial_loss = ResampleLoss(device=self.device, reweight_func='rebalance', loss_weight=1.0,reduction=reduction,
                                    focal=dict(focal=True, alpha=0.5, gamma=2),partial=True,
                                    logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                    map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                    class_freq=class_freq, train_num=train_num)
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.ffn(cls_hidden)

        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs, labels)
            elif train_type == 'pst_train':
                results["loss"] = self.partial_loss(outs, labels)           
            else:
                raise ValueError(f'not define train_type {train_type}')
        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.ffn(cls_hidden)

        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        return results


# TextCNN+CBLOSS = PST模型
class PSTModel(nn.Module):
    def __init__(
        self,
        device: int,
        class_freq: list,
        train_num: int,
        plm_filepath: str,
        num_filters: Optional[int] = 300,
        num_classes: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        kernel_sizes=[1, 3, 5],
        mid_dims=[
            100,
        ],
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        
        self.cnn = MultiKernelCNN(
            in_channel=hidden_size,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.ffn = FFN(
            input_dim=num_filters * len(kernel_sizes),
            output_dim=num_classes,
            mid_dims=mid_dims,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )
            
        self.loss = ResampleLoss(device=self.device, reweight_func='CB', loss_weight=5.0,reduction=reduction,
                                focal=dict(focal=True, alpha=0.5, gamma=2),partial=False,
                                logit_reg=dict(),
                                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                class_freq=class_freq, train_num=train_num) 
        self.partial_loss = ResampleLoss(device=self.device, reweight_func='CB', loss_weight=5.0,reduction=reduction,
                                focal=dict(focal=True, alpha=0.5, gamma=2),partial=True,
                                logit_reg=dict(),
                                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                class_freq=class_freq, train_num=train_num)      
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs, labels)
            elif train_type == 'pst_train':
                results["loss"] = self.partial_loss(outs, labels)
            else:
                raise ValueError(f'not define train_type {train_type}')

        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        return results


# FLEM
def kaiming_normal_init_net(net):
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def cosine_similarity(x, y):
    '''
    Cosine Similarity of two tensors
    Args:
        x: torch.Tensor, m x d
        y: torch.Tensor, n x d
    Returns:
        result, m x n
    '''
    assert x.size(1) == y.size(1)
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return x @ y.transpose(0, 1)

class LE(nn.Module):
    def __init__(self, num_feature, num_classes, hidden_dim=128):
        super(LE, self).__init__()
        self.fe1 = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fe2 = nn.Linear(hidden_dim, hidden_dim)
        self.le1 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.le2 = nn.Linear(hidden_dim, hidden_dim)
        self.de1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_classes),
        )
        self.de2 = nn.Linear(num_classes, num_classes)

    def forward(self, x, y):
        x = self.fe1(x) + self.fe2(self.fe1(x))
        y = self.le1(y) + self.le2(self.le1(y))
        d = torch.cat([x, y], dim=-1)
        d = self.de1(d) + self.de2(self.de1(d))
        return d

class FLEM(nn.Module):
    def __init__(
        self, 
        device: int,
        num_classes: int, 
        plm_filepath: str,
        le_hidden_dim: int,
        method='ld', 
        alpha=0.01, 
        beta=0.01,
        le_threshold=0,
        threshold=0.5):
        super().__init__()
        
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.le_threshold = le_threshold
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.le = LE(hidden_size, num_classes, le_hidden_dim)
        kaiming_normal_init_net(self.classifier)
        kaiming_normal_init_net(self.le)
        
        self.loss = self.forward_loss

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.classifier(cls_hidden)
        
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            labels = labels.to(torch.float)
            if train_type == 'base_train':
                le = self.le(cls_hidden, labels)
                results["loss"] = self.loss(outs, le, cls_hidden, labels, partial=False)
            elif train_type == 'pst_train':
                le = self.le(cls_hidden, labels)
                results["loss"] = self.loss(outs, le, cls_hidden, labels, partial=True)
            else:
                raise ValueError(f'not define train_type {train_type}')            
        return results

    def forward_loss(self, pred, le, x=None, y=None, partial=False):
        assert y is not None
        if partial:
            pseudo_label = y.masked_fill(y == 2, 1.0)
            pseudo_loss = self.loss_func(pred, pseudo_label)
            loss_pred_cls = pseudo_loss.masked_fill(y == 2, 0.0)
            loss_pred_cls = loss_pred_cls.mean()
        else:
            loss_pred_cls = self.loss_func(pred, y)
            loss_pred_cls = loss_pred_cls.mean()
        loss_pred_ld = nn.CrossEntropyLoss()(pred, torch.softmax(le.detach(), dim=1))
        loss_le_cls = self.loss_enhanced(le, pred, y)
        if self.method == 'ld':
            loss_le_spec = nn.CrossEntropyLoss()(le, torch.softmax(pred.detach(), dim=1))
        elif self.method == 'threshold':
            loss_le_spec = 0
            for i in range(pred.shape[0]):
                neg_index = y[i] == 0
                pos_index = y[i] == 1
                if torch.sum(pos_index) == 0: continue
                loss_le_spec += torch.maximum(le[i][neg_index][torch.argmax(le[i][neg_index])] - le[i][pos_index][
                    torch.argmin(le[i][pos_index])] + self.le_threshold, torch.tensor([0]).to(self.device))
        elif self.method == 'sim':
            with torch.no_grad():
                sim_x = cosine_similarity(x, x).detach()  # [n,n]
            sim_y = cosine_similarity(le, le)
            loss_le_spec = nn.MSELoss()(sim_y, sim_x)
        else:
            raise ValueError('Wrong E2ELE method!')
        loss_pred = self.alpha * loss_pred_ld + (1 - self.alpha) * loss_pred_cls
        loss_le = self.beta * loss_le_spec + (1 - self.beta) * loss_le_cls
        return loss_le + loss_pred

    def loss_enhanced(self, pred, teach, y):
        eps = 1e-7
        gamma1 = 0
        gamma2 = 1
        x_sigmoid = torch.sigmoid(pred)
        los_pos = y * torch.log(x_sigmoid.clamp(min=eps, max=1 - eps))
        los_neg = (1 - y) * torch.log((1 - x_sigmoid).clamp(min=eps, max=1 - eps))
        loss = los_pos + los_neg
        with torch.no_grad():
            teach_sigmoid = torch.sigmoid(teach)
            teach_pos = teach_sigmoid
            teach_neg = 1 - teach_sigmoid
            pt0 = teach_pos * y
            pt1 = teach_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = gamma1 * y + gamma2 * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w
        return -loss.mean()
    
    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_hidden = plm_outs.pooler_output
        outs = self.classifier(cls_hidden)
        
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}
            
        return results


# HTTN
class HTTN(nn.Module):
    def __init__(
        self,
        device: int,
        plm_filepath: str,
        num_filters: Optional[int] = 300,
        num_classes_head: Optional[int] = 173,
        num_classes_all: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        kernel_sizes=[1, 3, 5],
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        hidden_size = self.plm.config.hidden_size
        
        self.cnn = MultiKernelCNN(
            in_channel=hidden_size,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.liner = nn.Linear(num_filters * len(kernel_sizes), 128, bias=True)

        self.head = nn.Linear(128, num_classes_head, bias=False)
        self.all = nn.Linear(128, num_classes_all, bias=False)
        self.transfor = nn.Linear(128, 128, bias=False)
        
        self.head_loss = nn.BCELoss()
        self.transfor_loss = nn.MSELoss()
        self.threshold = threshold
    
    def extractor(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        cnn_outs = self.cnn(plm_hidden)
        extract = self.liner(cnn_outs)
        return extract
    
    def forward(self, extract, labels, train_head, train_transfor, **kwargs):        
        if train_head and labels is not None:
            head_outs = self.head(extract)
            labels = labels.to(torch.float)
            head_loss = self.head_loss(head_outs.sigmoid(), labels)
            return head_loss
        if train_transfor:
            real_weight=self.head.weight.data
            output = self.transfor(extract)
            transfor_loss = self.transfor_loss(output, real_weight)
            return transfor_loss
    
    def predict_head(self, input_ids, attention_mask, token_type_ids, **kwargs):
        extract = self.extractor(input_ids, attention_mask, token_type_ids)
        head_outs = self.head(extract)
        preds_head = head_outs.sigmoid().gt(self.threshold).long()
        logits_head = head_outs.sigmoid()
        results = {"logits": logits_head, "preds": preds_head}
        return results
    
    def build_all(self, extract_tail):
        tail_real = self.transfor(extract_tail)
        all_weight = torch.cat([self.head.weight.data, tail_real])
        self.all.weight.data = all_weight
        return
    
    def predict_all(self, input_ids, attention_mask, token_type_ids, **kwargs):
        extract = self.extractor(input_ids, attention_mask, token_type_ids)
        all_outs = self.all(extract)
        preds_all = all_outs.sigmoid().gt(self.threshold).long()
        logits_all = all_outs.sigmoid()
        results = {"logits": logits_all, "preds": preds_all}
        return results


# LSAN
class LSAN(nn.Module):
    def __init__(
        self,
        labels:list,
        device: int,
        plm_filepath: str,
        lstm_hid_dim: Optional[int] = 300,
        num_classes: Optional[int] = 173,
        d_a: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        self.tokenizer = BertTokenizerFast.from_pretrained(plm_filepath)
        self.n_classes = num_classes

        self.lstm = torch.nn.LSTM(self.plm.config.hidden_size, hidden_size=lstm_hid_dim, num_layers=1,
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.label_liner = torch.nn.Linear(768, lstm_hid_dim)
        
        self.linear_first = FFN(
            input_dim=lstm_hid_dim * 2,
            output_dim=num_classes,
            mid_dims=[d_a],
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )

        self.output_layer = FFN(
            input_dim=lstm_hid_dim * 2,
            output_dim=num_classes,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )
        self.lstm_hid_dim = lstm_hid_dim
        
        self.loss = nn.BCELoss(reduction=reduction)
        self.partial_loss = nn.BCELoss(reduction='none')
        self.threshold = threshold

        self.labels_input_ids, self.labels_attention_mask, self.labels_token_type_ids = self.init_label_embed(labels)

    def init_label_embed(self, labels):
        labels_input_ids = []
        labels_attention_mask = []
        labels_token_type_ids = []
        for label in labels:
            tokenized = self.tokenizer.encode_plus(
                text=label,
                add_special_tokens=True,
                padding="max_length",
                max_length=20,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized["token_type_ids"]
            labels_input_ids.append(input_ids)
            labels_attention_mask.append(attention_mask)
            labels_token_type_ids.append(token_type_ids)
        labels_input_ids = torch.tensor(labels_input_ids, dtype=torch.long, device=self.device)
        labels_attention_mask = torch.tensor(labels_attention_mask, dtype=torch.long, device=self.device)
        labels_token_type_ids = torch.tensor(labels_token_type_ids, dtype=torch.long, device=self.device)

        return labels_input_ids, labels_attention_mask, labels_token_type_ids

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        #step1 get LSTM outputs
        plm_hidden = plm_outs.last_hidden_state
        batch_size = input_ids.shape[0]
        hidden_state = (torch.randn(2,batch_size,self.lstm_hid_dim).cuda(device=self.device),torch.randn(2,batch_size,self.lstm_hid_dim).cuda(device=self.device))
        outputs, hidden_state = self.lstm(plm_hidden, hidden_state)
        #step2 get self-attention
        selfatt = self.linear_first(outputs)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt= selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)   
        #step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :,self.lstm_hid_dim:]
        plm_outs = self.plm(
            input_ids=self.labels_input_ids,
            attention_mask=self.labels_attention_mask,
            token_type_ids=self.labels_token_type_ids,
            return_dict=True,
        )
        labels_embed = plm_outs.pooler_output
        labels_embed = self.label_liner(labels_embed)
        m1 = torch.bmm(labels_embed.expand(batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(labels_embed.expand(batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att= torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)

        weight1=torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att ))
        weight1 = weight1/(weight1+weight2)
        weight2= 1-weight1

        doc = weight1*label_att+weight2*self_att
        avg_sentence_embeddings = torch.sum(doc, 1)/self.n_classes
        outs = self.output_layer(avg_sentence_embeddings)
    
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs.sigmoid(), labels.float())
            elif train_type == 'pst_train':
                # partial loss
                pseudo_label = labels.masked_fill(labels == 2, 1.0)
                pseudo_loss = self.partial_loss(outs.sigmoid(), pseudo_label.float())
                masked_loss = pseudo_loss.masked_fill(labels == 2, 0.0)
                results["loss"] = masked_loss.mean()
            else:
                raise ValueError(f'not define train_type {train_type}')
        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        #step1 get LSTM outputs
        plm_hidden = plm_outs.last_hidden_state
        batch_size = input_ids.shape[0]
        hidden_state = (torch.randn(2,batch_size,self.lstm_hid_dim).cuda(device=self.device),torch.randn(2,batch_size,self.lstm_hid_dim).cuda(device=self.device))
        outputs, hidden_state = self.lstm(plm_hidden, hidden_state)
        #step2 get self-attention
        selfatt = self.linear_first(outputs)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt= selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)   
        #step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :,self.lstm_hid_dim:]
        plm_outs = self.plm(
            input_ids=self.labels_input_ids,
            attention_mask=self.labels_attention_mask,
            token_type_ids=self.labels_token_type_ids,
            return_dict=True,
        )
        labels_embed = plm_outs.pooler_output
        labels_embed = self.label_liner(labels_embed)
        m1 = torch.bmm(labels_embed.expand(batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(labels_embed.expand(batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att= torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)

        weight1=torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att ))
        weight1 = weight1/(weight1+weight2)
        weight2= 1-weight1

        doc = weight1*label_att+weight2*self_att
        avg_sentence_embeddings = torch.sum(doc, 1)/self.n_classes
        outs = self.output_layer(avg_sentence_embeddings)
    
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}
        return results
    

# LACO
class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context

class LACO(nn.Module):
    def __init__(
        self,
        device: int,
        plm_filepath: str,
        vocab_size:int,
        num_filters: Optional[int] = 300,
        num_classes: Optional[int] = 173,
        dropout: Optional[float] = 0.5,
        threshold: float = 0.3,
        kernel_sizes=[1, 3, 5],
        mid_dims=[
            100,
        ],
        reduction = "mean",
    ):
        super().__init__()
        self.device = device
        self.plm = BertModel.from_pretrained(plm_filepath)
        self.plm.resize_token_embeddings(vocab_size)
        hidden_size = self.plm.config.hidden_size
        self.attention = selfAttention(8, hidden_size, hidden_size)
        self.cnn = MultiKernelCNN(
            in_channel=hidden_size,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.ffn = FFN(
            input_dim=num_filters * len(kernel_sizes),
            output_dim=num_classes,
            mid_dims=mid_dims,
            dropout=dropout,
            act_fn=nn.LeakyReLU(),
        )
        self.loss = nn.BCELoss(reduction=reduction)
        self.partial_loss = nn.BCELoss(reduction='none')
        self.threshold = threshold

    def forward(self, input_ids, attention_mask, token_type_ids, labels, train_type, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        plm_hidden = self.attention(plm_hidden)
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        if labels is not None:
            if train_type == 'base_train':
                results["loss"] = self.loss(outs.sigmoid(), labels.float())
            elif train_type == 'pst_train':
                # partial loss
                pseudo_label = labels.masked_fill(labels == 2, 1.0)
                pseudo_loss = self.partial_loss(outs.sigmoid(), pseudo_label.float())
                masked_loss = pseudo_loss.masked_fill(labels == 2, 0.0)
                results["loss"] = masked_loss.mean()
            else:
                raise ValueError(f'not define train_type {train_type}')
        return results

    def eval_pred(self, input_ids, attention_mask, token_type_ids, **kwargs):
        plm_outs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        plm_hidden = plm_outs.last_hidden_state
        plm_hidden = self.attention(plm_hidden)
        cnn_outs = self.cnn(plm_hidden)
        outs = self.ffn(cnn_outs)
        preds = outs.sigmoid().gt(self.threshold).long()
        logits = outs.sigmoid()
        results = {"logits": logits, "preds": preds}

        return results
   
