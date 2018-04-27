# coding=utf-8

import os
import csv
import cv2
import numpy as np
import pickle

parent_path = os.path.dirname(os.getcwd())
dataset_path = parent_path + "/dataset/"
train_data_path = parent_path + "/dataset/train/"

train_data_file = open(dataset_path + "train.csv", 'w', encoding='utf-8')
label_file = open(dataset_path + "label.csv", 'w', encoding='utf-8')

### 预处理训练数据集
writer = csv.writer(train_data_file)
writer.writerow(['filename','label'])

label_writer = csv.writer(label_file)

for label in os.listdir(train_data_path):
    label_writer.writerow(label)

    image_path = train_data_path + label
    for filename in os.listdir(image_path):
        writer.writerow([filename, label])

train_data_file.close()
label_file.close()


### 准备训练数据集
train_data_file = open(dataset_path + "train.csv", 'r', encoding='utf-8')
train_data_reader = csv.reader(train_data_file)

train_data = []

for line in train_data_reader:

    image_name = line[0]
    label = line[1]

    if (image_name == 'filename') or (label == 'label'):
        print("discard first row")
        continue

    imgAbsPath = train_data_path + label + '/' + image_name
    image = cv2.imread(imgAbsPath)

    resized_image = cv2.resize(image, (128, 128))

    normed_im = np.array([(resized_image - 127.5) / 127.5])

    train_data.append([label, normed_im])

train_data_file.close()

pickle.dump(train_data, open('train_data.dat','wb'))