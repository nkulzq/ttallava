import json

ann_ori = json.load(open("/home/wuyinjun/lzq/roco/ann_validation.json", 'r'))
ann = [{"id":item['image'].split('/')[-1].strip('.jpg').split('_')[-1], "image":item['image'], "conversations":[{"from":"human", "value":"<image>\nDescribe this medical image in detail."}, {"from":"gpt", "value":item['caption']}]} for item in ann_ori]
json.dump(ann, open("./ann_validation.json", 'w'))