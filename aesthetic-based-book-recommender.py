import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification, AutoTokenizer, AutoModelForSequenceClassification
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
from scipy.spatial.distance import cosine
import cv2

# Created by Jasper van der Valk (UvAnetID: 13854577) for Bachelor Thesis: Matching atmospheric descriptors in image and text: What are internet aesthetics anyway?


# ## Load Model and Dataset ## #
# ## ---------------------- ## #

# Load different parts of the dataset
ds_train = datasets.ImageFolder(root='/home/jaspervdvalk/Documents/Bachelor_Thesis/aesthetic-dataset/train')
ds_valid = datasets.ImageFolder(root='/home/jaspervdvalk/Documents/Bachelor_Thesis/aesthetic-dataset/valid')
ds_test = datasets.ImageFolder(root='/home/jaspervdvalk/Documents/Bachelor_Thesis/aesthetic-dataset/test')

# Load The Aesthetic Classifier Model (Visual Transformer)
num_classes = len(ds_train.classes)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# Model = V2: 81% Validation Accuracy
model = ViTForImageClassification.from_pretrained('PATH_TO_FILE/model10epochs', num_labels=num_classes)

# Load the classified books dataset
books_classified = pd.read_json("PATH_TO_FILE\merged-dataset-with-scores.json", lines=True)


# ## Evaluate the model on the Validation and Test Sets ## #
# ## -------------------------------------------------- ## #
loader_valid = DataLoader(ds_test, 32, shuffle=False)
loader_test = DataLoader(ds_test, 32, shuffle=False)

# Evaluation Function
def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            inputs = processor(images=list(images), return_tensors='pt')
            outputs = model(**inputs)

            predictions = outputs.logits.argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


# Get attention (heat) maps
def get_attention_weights(model, inputs):
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    return attentions


def process_attention_weights(attention_weights):
    # Stack and average the attention weights across all heads to get a good heatmap
    attention_weights = torch.stack(attention_weights).squeeze(1)
    attention_weights = torch.mean(attention_weights, dim=1)

    # Add residual connections and re-normalise
    residual = torch.eye(attention_weights.size(1))
    aug_att_mat = attention_weights + residual
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    return joint_attentions


# ## Then upload an image and the model will classify it: ## #
# ## ---------------------------------------------------- ## #
model.eval()

# Process the image
root = tk.Tk()
root.withdraw()
image_file_path = filedialog.askopenfilename(title="Select an Image")
if not image_file_path:
    print("No file selected.")
    exit()

test_image = Image.open(image_file_path).convert("RGB")
inputs = processor(images=test_image, return_tensors='pt')

# Get attention weights
attentions = get_attention_weights(model, inputs)
#print(attentions)

# Visualise Attention Maps
joint_attentions = process_attention_weights(attentions)
v = joint_attentions[-1]
grid_size = int(np.sqrt(joint_attentions.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = (mask - mask.min()) / (mask.max() - mask.min())
mask = cv2.resize(mask / mask.max(), test_image.size)[..., np.newaxis]
result = (mask * np.array(test_image)).astype("uint8")

# Apply colourmap to the mask
cmap = cm.get_cmap('jet')
heatmap = cmap(mask.squeeze())[:, :, :3]
heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(test_image.size, Image.BILINEAR))

result = (0.6 * np.array(test_image) + 0.4 * heatmap).astype(np.uint8)

# Get the output of the model
with torch.no_grad():
    outputs = model(**inputs)

class_names = ds_train.classes
predictions = outputs.logits.argmax(dim=1).item()

# Apply softmax function to normalise the logits (weights)
logits = outputs.logits.numpy().flatten()
exp_logits = np.exp(logits)
softmax = exp_logits / np.sum(exp_logits)

# Convert them to percentages
percentages = softmax
predicted_class_name = class_names[predictions]

# Display the prediction (output) of the model
print("Outputs in Percentages:")
print(class_names)
print(percentages)
print(f"The model predicts the image is: {predicted_class_name}\n")

closest_score = None
closest_distance = float('inf')
closest_books = []

for index, row in books_classified.iterrows():
    if "classifier_scores" in row:
        book_scores = np.array(row["classifier_scores"])
        percentages = np.array(percentages)
        distance = cosine(percentages, book_scores)

        # Create and maintain a list of most similar books
        if len(closest_books) < 10:
            closest_books.append((distance, row))
            closest_books.sort(key=lambda x: x[0]) # sort by distance
        else:
            if distance < closest_books[-1][0]:
                closest_books[-1] = (distance, row)
                closest_books.sort(key=lambda x: x[0]) # sort by distance

print("The Top 10 Books with the Closest Aesthetic to the Image are:\n")
for distance, book in closest_books:
    print(f"Similarity: {distance:.3f}")
    print(f"Title: {book['title']}")
    print(f"Classes: {book['classifier_labels']}")
    print(f"Aesthetic Score: {book['classifier_scores']}")
    print(f"Description: {book['description']}")
    print(f"Goodreads Link: {book['link']}")
    print("-------------------------------------------------------------------------\n")

# Display Image and Heatmap side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original Image
ax[0].imshow(test_image)
ax[0].set_title(f"Prediction: {predicted_class_name}")
ax[0].axis('off')

# Attention Heatmap
ax[1].imshow(result)
ax[1].set_title("Attention Heatmap")
ax[1].axis('off')

plt.show()
exit()
