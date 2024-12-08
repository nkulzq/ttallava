from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
import os
from kmeans_pytorch import kmeans
from transformers import BertTokenizer, BertModel
from torchclustermetrics import silhouette
from itertools import chain
import torch

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe the image."

ann = json.load(open("/home/wuyinjun/lzq/roco/ann_validation.json"))
image_files = [os.path.join("/home/wuyinjun/lzq/roco", item["image"]) for item in ann[:1024]]
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_files,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
captions, image_embeds = eval_model(args)

ids = [int(item.split('/')[-1].strip('.jpg').split('_')[-1]) for item in image_files]
json.dump(captions, open("./captions.json", 'w'))
image_embeds = [list(chain.from_iterable(item)) for item in image_embeds]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
caption_embeds = outputs.pooler_output
num_clusters = 3
labels, cluster_centers = kmeans(X=caption_embeds, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
group_size = 64
image_embeds = [image_embeds[i:i+group_size] for i in range(0, len(image_embeds), group_size)]
labels = [labels[i:i+group_size] for i in range(0, len(labels), group_size)]
captions = [captions[i:i+group_size] for i in range(0, len(captions), group_size)]
ids = [ids[i:i+group_size] for i in range(0, len(ids), group_size)]
caption_embeds = [caption_embeds[i:i+group_size] for i in range(0, len(caption_embeds), group_size)]
scores_image = []
scores_caption = []
print("scoring")
for image_embed, label in zip(image_embeds, labels):
    score = silhouette.score(image_embed, label)
    print(score)
    scores_image.append(score)
for image_embed, label in zip(caption_embeds, labels):
    score = silhouette.score(image_embed, label)
    print(score)
    scores_caption.append(score)
json.dump(scores_image, open("./scores_image.json", 'w'))
json.dump(scores_caption, open("./scores_caption.json", 'w'))
