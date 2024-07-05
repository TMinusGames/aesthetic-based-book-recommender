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

# Created by Jasper van der Valk (UvAnetID: 13854577) for Bachelor Thesis: Matching atmospheric descriptors in image and text: What are internet aesthetics anyway?


# Path to dataset
book_abstracts_chunk = pd.read_json(r'PATH_TO_FILE', lines=True, chunksize=1000000)
book_abstracts_top50000 = next(book_abstracts_chunk)
print('Abstracts Loaded')

# Change to drop the correct columns
columns_to_drop = [
    'user_id', 'review_id', 'rating', 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes', 'n_comments'
]

book_abstracts_top50000 = book_abstracts_top50000.drop(columns=columns_to_drop)

print('Dropped')

# Then save the new dataset as a json file to use in the application
book_abstracts_top50000.to_json('review-clean-small.json', orient='records', lines=True)
