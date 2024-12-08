from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

def score_eval(ref, gt):
    roco_eval = Scorer(ref, gt)
    result = roco_eval.compute_scores()
    return result

import json
import os

# captions = json.load(open("./captions_rocotta.json",'r'))
# captions = json.load(open("./captions_ori.json",'r'))
captions = json.load(open("./vicuna_tta_new.json",'r'))[:128]
annotation_file = json.load(open(os.path.join("/home/wuyinjun/lzq/roco","ann_validation.json"), 'r'))[:128]
gts = [item['caption'] for item in annotation_file]
results = []
ref = {i:[captions[i]] for i in range(len(captions))}
g = {i:[gts[i]] for i in range(len(gts))}
result = score_eval(ref, g)
# json.dump(results, open("./result.json", 'w'))