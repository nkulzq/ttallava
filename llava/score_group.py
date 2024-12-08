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

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe the image."

ann = json.load(open("/home/wuyinjun/lzq/roco/ann_validation.json"))
image_files = [os.path.join("/home/wuyinjun/lzq/roco", item["image"]) for item in ann][:2]
for i in range(1):
    image_file = image_files
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
    json.dump(captions, open("./captions.json", 'a'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    inputs = tokenizer(captions[0], return_tensors="pt")["input_ids"]
    get_embed = model.get_input_embeddings()
    group_size = 64
    with torch.no_grad():
        caption_embeds = get_embed(inputs)
    print(caption_embeds)
    num_clusters = 3
    labels, cluster_centers = kmeans(X=caption_embeds, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    image_embeds = [image_embeds[i:i+group_size] for i in range(0, len(image_embeds), group_size)]
    captions = [captions[i:i+group_size] for i in range(0, len(captions), group_size)]
    ids = [ids[i:i+group_size] for i in range(0, len(ids), group_size)]
    scores_image = []
    print("scoring")
    for image_embed, label in zip(image_embeds, labels):
        score = silhouette.score(image_embed, label)
        print(score)
        scores_image.append(score)
    json.dump(scores_image, open("./scores_image.json", 'a'))
