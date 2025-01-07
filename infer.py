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

model_path = "./checkpoints/llava-v1.5-13b-lora-tta-vicuna-new"
model_base = "liuhaotian/llava-v1.5-13b"
prompt = "Describe the picture in detail."

ann = json.load(open("/home/wuyinjun/lzq/roco/ann_validation.json"))
image_files = [os.path.join("/home/wuyinjun/lzq/roco", item["image"]) for item in ann][:128]
captions = [item["caption"] for item in ann]
args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_files,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "model": None
})()
outputs,image_embed = eval_model(args)
print(outputs)

json.dump(outputs, open("./captions.json", 'w'))
json.dump(image_embed, open("./image_embeds.json", 'w'))
