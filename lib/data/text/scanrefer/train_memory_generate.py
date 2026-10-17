'''
This is the script we propose for generating interaction-domain memory data during vanilla training. 
For each description in a scene, we randomly sample three other descriptions from the same scene and concatenate them with the original description to form a new description and memory field. 
You can adopt more advanced methods or leverage large models to obtain better interaction-domain memory generation during training.
'''
import json
import random

input_path = "./lib/data/text/scanrefer/ori_data.jsonl"
output_path = "./lib/data/text/scanrefer/train_data.jsonl"

data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

scene_groups = {}
for item in data:
    scene_groups.setdefault(item["scene_id"], []).append(item)

new_data = []
for scene_id, items in scene_groups.items():
    for item in items:
        candidates = [x["description"] for x in items]
        sampled = random.sample(candidates, 3)  
        random.shuffle(sampled)  
        
        new_description = item["description"] + " " + " ".join(sampled)
        
        new_item = dict(item)
        new_item["description"] = new_description
        new_item["memory"] = new_description
        
        new_data.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

