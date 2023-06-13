import re
import copy
from omegaconf import OmegaConf
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup
from rex.utils.logging import logger
from rex.utils.io import load_line_json
from rex.data.dataset import CachedDataset
from rex.data.manager import CachedManager
from rex.utils.tensor_move import move_to_device
from rex.utils.progress_bar import tqdm
from rex.tasks.base_task import TaskBase
from collections import defaultdict

from src.data.classification import TextClassificationTransform, HTTNTransform, LACOTransform
from src.model.classification import TextCNN, BertCLS, SetLoss, PSTModel, HTTN, FLEM, LSAN, LACO
from src.utils.utils import mcml_prf1, classification_auc, classification_p_at_k, classification_mean_average_p_at_k, evaluation, ndcg_k

class SelfTrainingTextClassificationTask(TaskBase):
    def __init__(self, config: OmegaConf, **kwargs) -> None:
        super().__init__(config, **kwargs)
        tokenizer = BertTokenizerFast.from_pretrained(config.plm_dir)
        if config.model_type == 'laco':
            self.transform = LACOTransform(tokenizer, self.config.max_seq_len, self.config.label2id_filepath, use_partial=False)
            self.partial_transform = LACOTransform(tokenizer, self.config.max_seq_len, self.config.label2id_filepath, use_partial=True)
        else:
            self.transform = TextClassificationTransform(tokenizer, self.config.max_seq_len, self.config.label2id_filepath, use_partial=False)
            self.partial_transform = TextClassificationTransform(tokenizer, self.config.max_seq_len, self.config.label2id_filepath, use_partial=True)
        self.data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            self.transform.collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=config.load_train_data,
            load_dev_data=config.load_dev_data,
            load_test_data=config.load_test_data,
        )

        # CB Loss
        label2count = defaultdict(int)
        for label in self.transform.label_encoder.label2id:
            label2count[label] = 0
        class_freq = []
        train_data = load_line_json(config.train_filepath)
        train_num = len(train_data)
        for data in train_data:
            labels = data['labels']
            for label in labels:
                label2count[label] += 1
        for val in label2count.values():
            class_freq.append(val)
        
        # 定义模型
        if config.model_type == 'textcnn':
            self.model = TextCNN(
                device=config.device,
                plm_filepath=config.plm_dir,
                num_filters=config.num_filters,
                num_classes=self.transform.label_encoder.num_tags,
                dropout=config.dropout,
                threshold=config.threshold,
                kernel_sizes=config.kernel_sizes,
                mid_dims=config.mid_dims,
                reduction=config.reduction,
            )
        elif config.model_type == 'bertcls':
            self.model = BertCLS(
                device=config.device,
                plm_filepath=config.plm_dir,
                num_classes=self.transform.label_encoder.num_tags,
                dropout=config.dropout,
                threshold=config.threshold,
                mid_dims=config.mid_dims,
                reduction=config.reduction,
            )
        elif config.model_type == 'LOSS':
            self.model = SetLoss(
                device=config.device,
                class_freq=class_freq,
                train_num=train_num,
                plm_filepath=config.plm_dir,
                loss_func_name=config.loss_func_name,
                num_classes=self.transform.label_encoder.num_tags,
                dropout=config.dropout,
                threshold=config.threshold,
                mid_dims=config.mid_dims,
                reduction=config.reduction,
            )
        elif config.model_type == 'PST':
            self.model = PSTModel(
                device=config.device,
                class_freq=class_freq,
                train_num=train_num,
                plm_filepath=config.plm_dir,
                num_filters=config.num_filters,
                num_classes=self.transform.label_encoder.num_tags,
                dropout=config.dropout,
                threshold=config.threshold,
                kernel_sizes=config.kernel_sizes,
                mid_dims=config.mid_dims,
                reduction=config.reduction,
            )
        elif config.model_type == 'flem':
            self.model = FLEM(
                device=config.device,
                num_classes=self.transform.label_encoder.num_tags,
                plm_filepath=config.plm_dir,
                le_hidden_dim=config.le_hidden_dim,
                method='ld', 
                alpha=config.alpha, 
                beta=config.beta,
                le_threshold=config.le_threshold,
                threshold=config.threshold,
            )
        elif config.model_type == 'lsan':
            self.model = LSAN(
                labels=list(self.transform.label_encoder.label2id.keys()),
                device=config.device,
                plm_filepath=config.plm_dir,
                lstm_hid_dim=config.lstm_hid_dim,
                num_classes=self.transform.label_encoder.num_tags,
                d_a=config.d_a,
                dropout=config.dropout,
                threshold=config.threshold,
                reduction=config.reduction,
            )
        elif config.model_type == 'laco':
            self.model = LACO(
                device=config.device,
                plm_filepath=config.plm_dir,
                vocab_size=len(self.transform.tokenizer),
                num_filters=config.num_filters,
                num_classes=self.transform.label_encoder.num_tags,
                dropout=config.dropout,
                threshold=config.threshold,
                kernel_sizes=config.kernel_sizes,
                mid_dims=config.mid_dims,
                reduction=config.reduction,
            )
        else:
            raise ValueError('error model type')
        self.model.to(self.config.device)
        
        # 初始化数据
        # self train
        self.base_self_train_data = []
        # PST
        self.textId2label_count = defaultdict(dict)
        self.gold_data = load_line_json(self.config.train_filepath)
        for data in self.gold_data:
            data['partial_labels'] = []
            data['data_type'] = 'gold'
            label2count = defaultdict(int)
            for label in self.transform.label_encoder.label2id:
                label2count[label] = 0
            for label in data['labels']:
                label2count[label] = 2
            self.textId2label_count[data['textId']] = label2count
        self.unlabel_data = load_line_json(self.config.unlabel_filepath)
        # self.unlabel_data = []
        for data in self.unlabel_data:
            data['labels'] = []
            data['partial_labels'] = []
            data['data_type'] = 'unlabel'
            label2count = defaultdict(int)
            for label in self.transform.label_encoder.label2id:
                label2count[label] = 0
            self.textId2label_count[data['textId']] = label2count
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        if config.load_train_data:
            self.scheduler = self.init_lr_scheduler()

    def init_lr_scheduler(self):
        num_training_steps = (len(self.data_manager.train_loader) * self.config.num_epochs) 
        num_warmup_steps = math.floor(num_training_steps * self.config.warmup_proportion)
        return get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            )

    def train(self):
        for name, params in self.model.named_parameters():
            last_layer = len(self.model.plm.encoder.layer) - 1
            if re.search(f"plm\.(embeddings|encoder\.layer\.(?!{last_layer}))", name):
                params.requires_grad = False
        best_measures = 0
        for epoch_idx in range(self.config.num_epochs):   
            if epoch_idx != 0:
                for name, params in self.model.named_parameters():
                    params.requires_grad = True
                 
            self.optimizer.zero_grad()
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.data_manager.train_loader, desc=f"Teacher Train(e{epoch_idx})")
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.config.device)
                batch['train_type'] = 'base_train'
                result = self.model(**batch)
                loss = result["loss"]
                loss = loss / self.config.gradient_accumulation_steps
                loader.set_postfix({"loss": loss.item()})
                loss.backward()
                if (step+1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (step+1) % 1000 == 0:
                    dev_measures = self.eval("dev")
                    test_measures = self.eval("test")
                    is_best = False
                    dev_select_f1 = dev_measures["micro"]["f1"]
                    if dev_select_f1 > self.best_metric:
                        is_best = True
                        self.best_metric = dev_select_f1
                        best_measures = dev_measures
                        self.best_epoch = epoch_idx
                        self.no_climbing_cnt = 0
                    else:
                        self.no_climbing_cnt += 1
                    if is_best and self.config.save_best_ckpt:
                        self.save_ckpt("best", epoch_idx)
                    logger.info(
                        f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
                    )

            logger.info(loader)

            dev_measures = self.eval("dev")
            test_measures = self.eval("test")

            is_best = False
            dev_select_f1 = dev_measures["micro"]["f1"]
            if dev_select_f1 > self.best_metric:
                is_best = True
                self.best_metric = dev_select_f1
                best_measures = dev_measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"Teacher Train finished. Best Epoch: {self.best_epoch}, Dev: {best_measures}",
            )
        )
        
    def self_training(self):
        # 加载最佳teacher model
        store_dict = torch.load(self.config.best_eval_model_filepath, map_location=torch.device(self.config.device))
        self.model.load_state_dict(store_dict["model_state"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # 开始利用teacher model 进行self training 并更新teacher model
        logger.info('start self training')
        self.model.train()

        best_measures = 0
        for epoch_idx in range(self.config.self_training_num_epochs):
            # 1.teacher model 更新数据
            self.tag_unlabel_data_base()
            # 2.新数据再训练
            self.optimizer.zero_grad()
            logger.info(f"Epoch: {epoch_idx}/{self.config.self_training_num_epochs}")
            self.model.train()
        
            gold_set = CachedDataset(self.transform(self.base_self_train_data, debug=False))
            gold_loader = DataLoader(
                gold_set,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.transform.collate_fn,
            )
            
            loader = tqdm(gold_loader, desc=f"self Train(e{epoch_idx})")
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.config.device)
                batch['train_type'] = 'base_train'
                result = self.model(**batch)
                loss = result["loss"]
                loss = loss / self.config.gradient_accumulation_steps
                loader.set_postfix({"loss": loss.item()})
                loss.backward()
                if (step+1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (step+1) % 1000 == 0:
                    dev_measures = self.eval("dev")
                    test_measures = self.eval("test")
                    is_best = False
                    dev_select_f1 = dev_measures["micro"]["f1"]
                    if dev_select_f1 > self.best_metric:
                        is_best = True
                        self.best_metric = dev_select_f1
                        best_measures = dev_measures
                        self.best_epoch = epoch_idx
                        self.no_climbing_cnt = 0
                    else:
                        self.no_climbing_cnt += 1
                    if is_best and self.config.save_best_ckpt:
                        self.save_ckpt("best", epoch_idx)
                    logger.info(
                        f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
                    )

            logger.info(loader)

            dev_measures = self.eval("dev")
            test_measures = self.eval("test")

            is_best = False
            dev_select_f1 = dev_measures["micro"]["f1"]
            if dev_select_f1 > self.best_metric:
                is_best = True
                self.best_metric = dev_select_f1
                best_measures = dev_measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"Self Train finished. Best Epoch: {self.best_epoch}, Dev: {best_measures}",
            )
        )

    def pst_training(self):
        # 加载最佳teacher model
        store_dict = torch.load(self.config.best_eval_model_filepath, map_location=torch.device(self.config.device))
        self.model.load_state_dict(store_dict["model_state"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # 开始利用teacher model 进行self training 并更新teacher model
        logger.info('start pst training')
        best_measures = 0
        for epoch_idx in range(self.config.self_training_num_epochs):
            # 1.teacher model 更新数据
            self.tag_unlabel_data_pst()
            # 2.新数据再训练
            self.optimizer.zero_grad()
            logger.info(f"Epoch: {epoch_idx}/{self.config.self_training_num_epochs}")
            self.model.train()
            
            if self.config.partial:
                gold_set = CachedDataset(self.partial_transform(self.gold_data, debug=False))
                gold_loader = DataLoader(
                    gold_set,
                    batch_size=self.config.train_batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=self.partial_transform.collate_fn,
                )
            else:
                gold_set = CachedDataset(self.transform(self.gold_data, debug=False))
                gold_loader = DataLoader(
                    gold_set,
                    batch_size=self.config.train_batch_size,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=self.transform.collate_fn,
                )
            
            loader = tqdm(gold_loader, desc=f"pst Train(e{epoch_idx})")
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.config.device)
                batch['train_type'] = 'pst_train'
                result = self.model(**batch)
                loss = result["loss"]
                loss = loss / self.config.gradient_accumulation_steps
                loader.set_postfix({"loss": loss.item()})
                loss.backward()
                if (step+1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (step+1) % 1000 == 0:
                    dev_measures = self.eval("dev")
                    test_measures = self.eval("test")
                    is_best = False
                    dev_select_f1 = dev_measures["micro"]["f1"]
                    if dev_select_f1 > self.best_metric:
                        is_best = True
                        self.best_metric = dev_select_f1
                        best_measures = dev_measures
                        self.best_epoch = epoch_idx
                        self.no_climbing_cnt = 0
                    else:
                        self.no_climbing_cnt += 1
                    if is_best and self.config.save_best_ckpt:
                        self.save_ckpt("best", epoch_idx)
                    logger.info(
                        f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
                    )

            logger.info(loader)

            dev_measures = self.eval("dev")
            test_measures = self.eval("test")

            is_best = False
            dev_select_f1 = dev_measures["micro"]["f1"]
            if dev_select_f1 > self.best_metric:
                is_best = True
                self.best_metric = dev_select_f1
                best_measures = dev_measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"PST Train finished. Best Epoch: {self.best_epoch}, Dev: {best_measures}",
            )
        )

    def tag_unlabel_data_base(self):
        # teacher model 来给unlabel打分  self train 只要大于阈值的数据
        self.base_self_train_data = []
        self.base_self_train_data.extend(self.gold_data)
        self.model.eval()
        all_data = copy.deepcopy(self.unlabel_data)
        all_data_set = CachedDataset(self.transform.eval_transform(all_data))
        all_data_loader = DataLoader(
            all_data_set,
            batch_size=self.config.eval_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.transform.collate_fn,
        )
        all_data_loader = tqdm(all_data_loader, desc=f"updata unlabel data self train")
        pred_textId2sources = {}
        for raw_batch in all_data_loader:
            raw_batch = move_to_device(raw_batch, self.config.device)
            outputs = self.model.eval_pred(**raw_batch)
            batch_pred_source = outputs["logits"].detach().tolist()
            for source, textId in zip(batch_pred_source, raw_batch['textId']):
                pred_textId2sources[textId] = source

        all_data = tqdm(all_data, desc=f"change_data_by_source")
        for data in all_data:
            source = pred_textId2sources[data['textId']]
            pred_first = set()
            for i,one_score in enumerate(source):
                label = self.transform.label_encoder.id2label[i]
                if one_score >= self.config.threshold_first:
                    pred_first.add(label)
            data['labels'] = list(pred_first)
            if data['labels'] == []:
                data['data_type'] = 'unlabel'
            else:
                data['data_type'] = 'gold'
                self.base_self_train_data.append(data)                

        self.model.train()
        return

    def tag_unlabel_data_pst(self):
        # teacher model 来给unlabel打分  挑选高分去gold  中分进partial  低分还是unlabel
        self.model.eval()
        all_data = []
        all_data.extend(self.gold_data)
        all_data.extend(self.unlabel_data)
        self.gold_data = []
        self.unlabel_data = []
        all_data_set = CachedDataset(self.transform.eval_transform(all_data))
        all_data_loader = DataLoader(
            all_data_set,
            batch_size=self.config.eval_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.transform.collate_fn,
        )
        all_data_loader = tqdm(all_data_loader, desc=f"updata unlabel data PST")
        pred_textId2sources = {}
        for raw_batch in all_data_loader:
            raw_batch = move_to_device(raw_batch, self.config.device)
            outputs = self.model.eval_pred(**raw_batch)
            batch_pred_source = outputs["logits"].detach().tolist()
            for source, textId in zip(batch_pred_source, raw_batch['textId']):
                pred_textId2sources[textId] = source
        
        all_data = tqdm(all_data, desc=f"change_data_by_source")
        for data in all_data:
            label2count = self.textId2label_count[data['textId']]
            gold_labels = set()
            negative_labels = set()
            for label in label2count:
                if label2count[label] >= 2:
                    gold_labels.add(label)
                if label2count[label] <= -2:
                    negative_labels.add(label)

            source = pred_textId2sources[data['textId']]

            pred_first = set()
            pred_second = set()
            for i,one_score in enumerate(source):
                label = self.transform.label_encoder.id2label[i]
                if label2count[label] >= 2 or label2count[label] <= -2:
                    continue
                if one_score >= self.config.threshold_first:
                    label2count[label] += 1
                    pred_first.add(label)
                elif self.config.threshold_second <= one_score < self.config.threshold_first:
                    pred_second.add(label)
                else:
                    label2count[label] -= 1
            self.textId2label_count[data['textId']] = label2count

            labels = list(gold_labels | pred_first)
            data['labels'] = labels
            partial_labels = list(pred_second)
            data['partial_labels'] = partial_labels
            
            if data['labels'] != []:
                data['data_type'] = 'gold'
                self.gold_data.append(data)
            else:
                if data['partial_labels'] != []:
                    if self.config.partial:
                        data['data_type'] = 'partial'
                        self.gold_data.append(data)
                    else:
                        data['data_type'] = 'unlabel'
                        self.unlabel_data.append(data)  
                else:
                    data['data_type'] = 'unlabel'
                    self.unlabel_data.append(data)
        self.model.train()
        return
    
    @torch.no_grad()
    def eval(self, dataset_name: str):
        self.model.eval()
        name2loader = {
            "train": self.data_manager.train_eval_loader,
            "dev": self.data_manager.dev_loader,
            "test": self.data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval")
        preds = []
        pred_sources = []
        golds = []
        for raw_batch in loader:
            raw_batch = move_to_device(raw_batch, self.config.device)
            golds.extend(raw_batch["labels"].cpu().tolist())
            outputs = self.model.eval_pred(**raw_batch)
            batch_pred_ = outputs["preds"].detach().tolist()
            batch_pred_source = outputs["logits"].detach().tolist()
            pred_sources.extend(batch_pred_source)
            preds.extend(batch_pred_)
        logger.info(loader)
        result_measures = mcml_prf1(preds, golds, self.transform.label_encoder.id2label)
        auc = classification_auc(np.array(golds), np.array(pred_sources))
        p_at_1 = classification_p_at_k(golds, pred_sources, 1, 0)
        p_at_3 = classification_p_at_k(golds, pred_sources, 3, 0)
        p_at_5 = classification_p_at_k(golds, pred_sources, 5, 0)
        map_at_k =  classification_mean_average_p_at_k(golds, pred_sources)
        res = evaluation(golds, pred_sources, preds)
        ndk = ndcg_k(preds, golds)
        result_measures['auc'] = auc
        result_measures['p_at_1'] = p_at_1
        result_measures['p_at_3'] = p_at_3
        result_measures['p_at_5'] = p_at_5
        result_measures['map_at_k'] = map_at_k
        result_measures['Hamming'] = res['Hamming']
        result_measures['Ranking'] = res['Ranking']
        result_measures['Coverage'] = res['Coverage']
        result_measures['mAP'] = res['mAP']
        result_measures['ndk1'] = ndk[0]
        result_measures['ndk3'] = ndk[1]
        result_measures['ndk5'] = ndk[2]
        self.model.train()
        return result_measures

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        data_type = []
        for d in tqdm(data, desc='textclassification'):
            new_d = copy.deepcopy(d)
            new_d["labels"] = []
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.eval_pred(**tensor_batch)
            preds = np.array(outputs["preds"].cpu().tolist()[0])
            pred_type_ids = np.array(np.where(preds==1)).tolist()[0]
            for type_id in pred_type_ids:
                type_label = self.transform.label_encoder.id2label[type_id]
                new_d["labels"].append(type_label)

            data_type.append(new_d)
        return data_type

    @torch.no_grad()
    def predict_batch(self, data):
        self.model.eval()
        data_type = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_d["labels"] = []
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.eval_pred(**tensor_batch)
            preds = np.array(outputs["preds"].cpu().tolist()[0])
            pred_type_ids = np.array(np.where(preds==1)).tolist()[0]
            for type_id in pred_type_ids:
                type_label = self.transform.label_encoder.id2label[type_id]
                new_d["labels"].append(type_label)
            data_type.append(new_d)
        return data_type

    @torch.no_grad()
    def get_probs(self, data):
        self.model.eval()
        data_probs = []
        for d in tqdm(data, desc='textclassification'):
            new_d = copy.deepcopy(d)
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.eval_pred(**tensor_batch)
            probs = outputs["logits"].cpu().tolist()[0]

            new_d["probs"] = probs
            data_probs.append(new_d)
        return data_probs
    
    @torch.no_grad()
    def get_probs_batch(self, data):
        self.model.eval()
        data_probs = []
        for d in data:
            new_d = copy.deepcopy(d)
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.eval_pred(**tensor_batch)
            probs = outputs["logits"].cpu().tolist()[0]

            new_d["probs"] = probs
            data_probs.append(new_d)
        return data_probs


class HTTNTask(TaskBase):
    def __init__(self, config: OmegaConf, **kwargs) -> None:
        super().__init__(config, **kwargs)
        tokenizer = BertTokenizerFast.from_pretrained(config.plm_dir)
        self.transform = HTTNTransform(tokenizer, self.config.max_seq_len, self.config.label2id_filepath, self.config.num_head_labels)

        self.head_data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedDataset,
            self.transform.head_transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            self.transform.collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=True,
            load_dev_data=True,
            load_test_data=True,
        )
        
        self.all_data_manager = CachedManager(
            config.train_filepath,
            config.dev_filepath,
            config.test_filepath,
            CachedDataset,
            self.transform,
            load_line_json,
            config.train_batch_size,
            config.eval_batch_size,
            self.transform.collate_fn,
            debug_mode=config.debug_mode,
            load_train_data=False,
            load_dev_data=True,
            load_test_data=True,
        )

        self.model = HTTN(
            device=config.device,
            plm_filepath=config.plm_dir,
            num_filters=config.num_filters,
            num_classes_head=config.num_head_labels,
            num_classes_all=config.num_all_labels,
            dropout=config.dropout,
            threshold=config.threshold,
            kernel_sizes=config.kernel_sizes,
        )

        self.model.to(self.config.device)
        
        self.transfor_train_data = load_line_json(self.config.train_filepath)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        if config.load_train_data:
            self.scheduler = self.init_lr_scheduler()

    def init_lr_scheduler(self):
        num_training_steps = (len(self.head_data_manager.train_loader) * self.config.num_epochs) 
        num_warmup_steps = math.floor(num_training_steps * self.config.warmup_proportion)
        return get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            )

    def train_head(self):
        for name, params in self.model.named_parameters():
            last_layer = len(self.model.plm.encoder.layer) - 1
            if re.search(f"plm\.(embeddings|encoder\.layer\.(?!{last_layer}))", name):
                params.requires_grad = False
        best_measures = 0
        for epoch_idx in range(self.config.num_epochs):   
            if epoch_idx != 0:
                for name, params in self.model.named_parameters():
                    params.requires_grad = True
                 
            self.optimizer.zero_grad()
            logger.info(f"Epoch: {epoch_idx}/{self.config.num_epochs}")
            self.model.train()
            loader = tqdm(self.head_data_manager.train_loader, desc=f"Head Train(e{epoch_idx})")
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, self.config.device)
                extract = self.model.extractor(**batch)
                loss = self.model(extract=extract, labels=batch['labels'], train_head=True, train_transfor=False)
                loss = loss / self.config.gradient_accumulation_steps
                loader.set_postfix({"loss": loss.item()})
                loss.backward()
                if (step+1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if (step+1) % 1000 == 0:
                    dev_measures = self.eval("dev_head")
                    test_measures = self.eval("test_head")
                    is_best = False
                    dev_select_f1 = dev_measures["micro"]["f1"]
                    if dev_select_f1 > self.best_metric:
                        is_best = True
                        self.best_metric = dev_select_f1
                        best_measures = dev_measures
                        self.best_epoch = epoch_idx
                        self.no_climbing_cnt = 0
                    else:
                        self.no_climbing_cnt += 1
                    if is_best and self.config.save_best_ckpt:
                        self.save_ckpt("best", epoch_idx)
                    logger.info(
                        f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
                    )

            logger.info(loader)

            dev_measures = self.eval("dev_head")
            test_measures = self.eval("test_head")

            is_best = False
            dev_select_f1 = dev_measures["micro"]["f1"]
            if dev_select_f1 > self.best_metric:
                is_best = True
                self.best_metric = dev_select_f1
                best_measures = dev_measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, step: {step}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"Train finished. Best Epoch: {self.best_epoch}, Dev: {best_measures}",
            )
        )

    def train_transfor(self):
        # 加载最佳teacher model
        store_dict = torch.load(self.config.best_eval_model_filepath, map_location=torch.device(self.config.device))
        self.model.load_state_dict(store_dict["model_state"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate*10)
        logger.info('start transfor train')
        self.best_metric = 0
        for epoch_idx in range(self.config.transfor_train_num_epochs):
            self.optimizer.zero_grad()
            logger.info(f"Epoch: {epoch_idx}/{self.config.transfor_train_num_epochs}")
            self.model.train()
            sample_head_datas = self.transform.sample_head_tail(self.config.sample_pre_num, self.config.sample_times, 'head', self.transfor_train_data)
            for i in range(self.config.sample_times):
                extrect = []
                sample_label2data = sample_head_datas[i]
                for _,data in sample_label2data.items():
                    gold_set = CachedDataset(self.transform.transform(data, debug=False))
                    gold_loader = DataLoader(
                        gold_set,
                        batch_size=self.config.sample_pre_num,
                        shuffle=True,
                        num_workers=1,
                        collate_fn=self.transform.collate_fn,
                    )
                    for batch in gold_loader:
                        batch = move_to_device(batch, self.config.device)
                        label_extrect = self.model.extractor(**batch)
                        label_extrect = torch.mean(label_extrect, dim=0).cpu().tolist()
                    extrect.append(label_extrect)
                extrect = torch.tensor(extrect)
                extrect = move_to_device(extrect, self.config.device)
                loss = self.model(extract=extrect, labels=None, train_head=False, train_transfor=True)
                logger.info({"loss": loss.item()})
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            extract_tail = []
            sample_label2data = self.transform.sample_head_tail(self.config.sample_pre_num, 1, 'tail', self.transfor_train_data)[0]
            for _,data in sample_label2data.items():
                gold_set = CachedDataset(self.transform.transform(data, debug=False))
                gold_loader = DataLoader(
                    gold_set,
                    batch_size=self.config.sample_pre_num,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=self.transform.collate_fn,
                )
                for batch in gold_loader:
                    batch = move_to_device(batch, self.config.device)
                    label_extrect = self.model.extractor(**batch)
                    label_extrect = torch.mean(label_extrect, dim=0).cpu().tolist()
                extract_tail.append(label_extrect)
            extract_tail = torch.tensor(extract_tail)
            extract_tail = move_to_device(extract_tail, self.config.device)
            self.model.build_all(extract_tail)
            dev_measures = self.eval("dev")
            test_measures = self.eval("test")    
            
            is_best = False
            dev_select_f1 = dev_measures["micro"]["f1"]
            if dev_select_f1 > self.best_metric:
                is_best = True
                self.best_metric = dev_select_f1
                best_measures = dev_measures
                self.best_epoch = epoch_idx
                self.no_climbing_cnt = 0
            else:
                self.no_climbing_cnt += 1

            if is_best and self.config.save_best_ckpt:
                self.save_ckpt("best", epoch_idx)

            logger.info(
                f"Epoch: {epoch_idx}, is_best: {is_best}, Dev: {dev_measures}, Test: {test_measures}"
            )

            if (
                self.config.num_early_stop > 0
                and self.no_climbing_cnt > self.config.num_early_stop
            ):
                break

        logger.info(
            (
                f"Train finished. Best Epoch: {self.best_epoch}, Dev: {best_measures}",
            )
        )

    @torch.no_grad()
    def eval(self, dataset_name: str):
        self.model.eval()
        name2loader = {
            "dev_head": self.head_data_manager.dev_loader,
            "test_head": self.head_data_manager.test_loader,
            "dev": self.all_data_manager.dev_loader,
            "test": self.all_data_manager.test_loader,
        }
        loader = tqdm(name2loader[dataset_name], desc=f"{dataset_name} Eval")
        preds = []
        pred_sources = []
        golds = []
        for raw_batch in loader:
            raw_batch = move_to_device(raw_batch, self.config.device)
            golds.extend(raw_batch["labels"].cpu().tolist())
            # predict_head
            if dataset_name.find('head') != -1:
                outputs = self.model.predict_head(**raw_batch)
            else:
                outputs = self.model.predict_all(**raw_batch)
            batch_pred_ = outputs["preds"].detach().tolist()
            batch_pred_source = outputs["logits"].detach().tolist()
            pred_sources.extend(batch_pred_source)
            preds.extend(batch_pred_)
        logger.info(loader)
        if dataset_name.find('head') != -1:
            id2label = self.transform.head_label_encoder.id2label
        else:
            id2label = self.transform.label_encoder.id2label
        result_measures = mcml_prf1(preds, golds, self.transform.label_encoder.id2label)
        auc = classification_auc(np.array(golds), np.array(pred_sources))
        p_at_1 = classification_p_at_k(golds, pred_sources, 1, 0)
        p_at_3 = classification_p_at_k(golds, pred_sources, 3, 0)
        p_at_5 = classification_p_at_k(golds, pred_sources, 5, 0)
        map_at_k =  classification_mean_average_p_at_k(golds, pred_sources)
        res = evaluation(golds, pred_sources, preds)
        ndk = ndcg_k(preds, golds)
        result_measures['auc'] = auc
        result_measures['p_at_1'] = p_at_1
        result_measures['p_at_3'] = p_at_3
        result_measures['p_at_5'] = p_at_5
        result_measures['map_at_k'] = map_at_k
        result_measures['Hamming'] = res['Hamming']
        result_measures['Ranking'] = res['Ranking']
        result_measures['Coverage'] = res['Coverage']
        result_measures['mAP'] = res['mAP']
        result_measures['ndk1'] = ndk[0]
        result_measures['ndk3'] = ndk[1]
        result_measures['ndk5'] = ndk[2]
        self.model.train()
        return result_measures

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        data_type = []
        for d in tqdm(data, desc='textclassification'):
            new_d = copy.deepcopy(d)
            new_d["labels"] = []
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.all_data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.predict_all(**tensor_batch)
            preds = np.array(outputs["preds"].cpu().tolist()[0])
            pred_type_ids = np.array(np.where(preds==1)).tolist()[0]
            for type_id in pred_type_ids:
                type_label = self.transform.label_encoder.id2label[type_id]
                new_d["labels"].append(type_label)

            data_type.append(new_d)
        return data_type

    @torch.no_grad()
    def predict_batch(self, data):
        self.model.eval()
        data_type = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_d["labels"] = []
            batch = self.transform.predict_transform(new_d)
            tensor_batch = self.all_data_manager.collate_fn([batch])
            tensor_batch = move_to_device(tensor_batch, self.config.device)
            outputs = self.model.predict_all(**tensor_batch)
            preds = np.array(outputs["preds"].cpu().tolist()[0])
            pred_type_ids = np.array(np.where(preds==1)).tolist()[0]
            for type_id in pred_type_ids:
                type_label = self.transform.label_encoder.id2label[type_id]
                new_d["labels"].append(type_label)
            data_type.append(new_d)
        return data_type
