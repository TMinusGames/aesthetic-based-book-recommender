import torch
import numpy as np
from torchvision import datasets, transforms
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_metric
from PIL import Image
import matplotlib.pyplot as plt

# Created by Jasper van der Valk (UvAnetID: 13854577) for Bachelor Thesis: Matching atmospheric descriptors in image and text: What are internet aesthetics anyway?


# Load different parts of the dataset
ds_train = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\train')
ds_valid = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\valid')
ds_test = datasets.ImageFolder(root=r'c:\Users\jalva\Documents\Bachelor_Thesis\aesthetic-dataset\test')

# Load pre-trained ViT Model
num_classes = len(ds_train.classes)
print(num_classes)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Adjust the final layer of the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)


def collate_fn(batch):
    images, labels = zip(*batch)
    # pixel_values = torch.stack([transform(image) for image in images])
    inputs = processor(images, return_tensors='pt')
    # labels = torch.tensor(labels)
    # return {'pixel-values': pixel_values, 'labels': labels}
    inputs['labels'] = torch.tensor(labels)
    return inputs


metric = load_metric("accuracy", trust_remote_code=True)


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


# Define training arguments
training_arguments = TrainingArguments(
    output_dir=r'c:\Users\jalva\Documents\Bachelor_Thesis',
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=5,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

# Then train the model
trainer = Trainer(
    model=model,
    args=training_arguments,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    tokenizer=processor
)

# # Results
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Evaluation
valid_metrics = trainer.evaluate(ds_valid)
trainer.log_metrics("eval", valid_metrics)
trainer.save_metrics("eval", valid_metrics)

test_metrics = trainer.evaluate(ds_test)
trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)
