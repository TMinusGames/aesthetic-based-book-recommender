import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
from datasets import load_metric
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# To use GPU instead of CPU
device = 0 if torch.cuda.is_available() else -1

# Created by Jasper van der Valk (UvAnetID: 13854577) for Bachelor Thesis: Matching atmospheric descriptors in image and text: What are internet aesthetics anyway?


# ## Load Model and Dataset ## #
# ## ---------------------- ## #

# Load different parts of the dataset
ds_train = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\train')
ds_valid = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\valid')
ds_test = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\test')

# Load The Text Classifier Model (Comprehend_it-base)
classifier = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base", device=device)
candidate_labels = ds_train.classes

# Load the json file for the combined dataset
merged_dataset = pd.read_json(r'PATH_TO_FILE\merged-dataset.json', lines=True)
print('Dataset Loaded')

# Inspect the dataset
print("Merged Dataset:")
print(merged_dataset)

# The Classification
classifier_labels = []
classifier_scores = []
counter = 0

for index, row in merged_dataset.iterrows():
    book_description = row['title'] + ': ' + row['description'] + ' ' + row['review_text']
    print(counter)
    output = classifier(book_description, candidate_labels, multi_label=True)

    labels = output['labels']
    scores = output['scores']

    # Sort the labels alphabetically so that they can be matched with the user image
    label_score_list = list(zip(labels, scores))
    label_score_list.sort(key=lambda x: x[0])

    # Separate sorted labels and score pairs
    labels_2, scores_2 = zip(*label_score_list)

    classifier_labels.append(list(labels_2))
    classifier_scores.append(list(scores_2))
    counter += 1

merged_dataset['classifier_labels'] = classifier_labels
merged_dataset['classifier_scores'] = classifier_scores

# Then save the new dataset as a json file to use in the application
merged_dataset.to_json('merged-dataset-with-scores.json', orient='records', lines=True)
