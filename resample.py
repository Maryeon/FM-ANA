import os
import os.path as osp
import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import defaultdict
from dataset import BaseDataset


label_file = "/path/to/labels.json"


def resample(labels, max_num_samples=1000):
    label_to_cnt = defaultdict(lambda: 0)
    label_to_img_idx = defaultdict(list)

    img_ids = list(labels.keys())
    flags = [0] * len(img_ids)
    for i, img_id in enumerate(img_ids):
        for label in labels[img_id]:
            label_to_cnt[label] += 1
            label_to_img_idx[label].append(i)

    label_names = list(label_to_cnt.keys())
    label_names = sorted(label_names, key=lambda id: label_to_cnt[id])
    
    new_labels = []
    new_label_to_cnt = defaultdict(lambda: 0)
    for label_name in label_names:
        available_img_ids = []
        for img_idx in label_to_img_idx[label_name]:
            if not flags[img_idx]:
                available_img_ids.append(img_ids[img_idx])
                flags[img_idx] = 1
        num_sample = min(max_num_samples, max(0, max_num_samples-new_label_to_cnt[label_name]))
        if num_sample > 0:
            if num_sample < len(available_img_ids):
                sampled_img_ids = random.sample(available_img_ids, num_sample)
            else:
                sampled_img_ids = available_img_ids
            
            for img_id in sampled_img_ids:
                new_labels.append({
                    'img': img_id+'.png',
                    'label': labels[img_id]
                })

                for label in labels[img_id]:
                    new_label_to_cnt[label] += 1

    return new_labels


def main():
    with open(label_file, "r") as f:
        labels = json.load(f)

    img_id_to_labels = {}
    for item in labels:
        if "AC-0" not in item["label"]:
            img_id_to_labels[item["img"].split(".")[0]] = item["label"]

    resample_labels = resample(img_id_to_labels)
    with open('/path/to/splits.json', 'w') as f:
        json.dump(resample_labels, f, indent=4)


if __name__ == '__main__':
    main()