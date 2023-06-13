# 人为漏标AAPD和RCV1-V2的训练集
from rex.utils.io import load_line_json, dump_line_json
from random import shuffle
import random
import copy
from tqdm import tqdm

def divide_train(dataset_name:str):
    def lose_fake_prob(train_data:list, fake_dir:str):
        # 假的漏标率，0.1-0.9
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for prob in tqdm(probs):
            new_datas = []
            for data in train_data:
                new_data = copy.deepcopy(data)
                new_labels = []
                for label in new_data['labels']:
                    random1 = random.random()
                    if random1 >= prob:
                        new_labels.append(label)
                new_data['labels'] = new_labels
                new_datas.append(new_data)
            dump_line_json(new_datas, f'{fake_dir}/train_{prob}.jsonl')
        return

    def lose_real_prob(train_data:list, real_dir:str, dataset_name:str):
        # 真实场景漏标率，有漏标率上限
        # AAPD: 0.585
        # RCV1-V2: 0.709
        if dataset_name == 'AAPD':
            probs = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif dataset_name == 'RCV1-V2':
            probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        else:
            raise ValueError(f'not find dataset_name: {dataset_name}')
        all_label_counts = 0
        for data in train_data:
            all_label_counts += len(data['labels'])
        for prob in tqdm(probs):
            shuffle(train_data)
            lose_label_counts = int(all_label_counts * prob)
            new_datas = copy.deepcopy(train_data)
            repet_count = 0
            while(lose_label_counts > 0):
                for data in new_datas:
                    if lose_label_counts <= 0:
                        break
                    if len(data['labels']) == 1:
                        continue
                    if random.random() <= prob + repet_count:
                        label = random.choice(data['labels'])
                        data['labels'].remove(label)
                        lose_label_counts -= 1
                repet_count += 0.01
            dump_line_json(new_datas, f'{real_dir}/train_{prob}.jsonl')
        return

    def lose_one(train_data:list, one_path:str):
        new_datas = []
        for data in tqdm(train_data):
            new_data = copy.deepcopy(data)
            label = random.choice(new_data['labels'])
            new_labels = [label]
            new_data['labels'] = new_labels
            new_datas.append(new_data)
        dump_line_json(new_datas, one_path)
        return

    train_path = f'data/{dataset_name}/raw_data/train.jsonl'
    train_data = load_line_json(train_path)
    fake_dir = f'data/{dataset_name}/lose_fake'
    lose_fake_prob(train_data, fake_dir)
    real_dir = f'data/{dataset_name}/lose_real'
    lose_real_prob(train_data, real_dir, dataset_name)
    one_path = f'data/{dataset_name}/raw_data/train_one.jsonl'
    lose_one(train_data, one_path)

divide_train('AAPD')
divide_train('RCV1-V2')

import os
def count_train(dataset_name:str):
    all_train = load_line_json(f'data/{dataset_name}/raw_data/train.jsonl')
    raw_label_counts = 0
    for data in all_train:
        raw_label_counts += len(data['labels'])
    fake_dir = f'data/{dataset_name}/lose_fake'
    for file_name in os.listdir(fake_dir):
        file_path = f'{fake_dir}/{file_name}'
        train_data = load_line_json(file_path)
        label_counts = 0
        for data in train_data:
            label_counts += len(data['labels'])
        print(f'dataset_name:{dataset_name}, fake: {file_name}:{label_counts/raw_label_counts}')
    real_dir = f'data/{dataset_name}/lose_real'
    for file_name in os.listdir(real_dir):
        file_path = f'{real_dir}/{file_name}'
        train_data = load_line_json(file_path)
        label_counts = 0
        for data in train_data:
            label_counts += len(data['labels'])
        print(f'dataset_name:{dataset_name}, real: {file_name}:{label_counts/raw_label_counts}')


count_train('AAPD')
count_train('RCV1-V2')