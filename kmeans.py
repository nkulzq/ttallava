from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
import os
from kmeans_pytorch import kmeans
from transformers import BertTokenizer, BertModel
from torchclustermetrics import silhouette
import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.cluster import KMeans

group_size = 32
captions = json.load(open("./captions_epoch1.json", 'r'))
image_embeds = json.load(open("./image_embeds_epoch1.json", 'r'))
image_embeds = [image_embeds[i:i+group_size] for i in range(0, len(image_embeds), group_size)]
labels = []
for image_embed in image_embeds:
    clustering = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(image_embed)
    label = clustering.labels_
    print(label)
    labels.extend(label.tolist())
json.dump(labels, open("./labels.json", 'w'))
