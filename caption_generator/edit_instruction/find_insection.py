# 把筛掉的找回来
import json
from tqdm import tqdm

def find_insection(type = 'action_change'):
    with open(f'./output/mscoco_{type}_-100.jsonl', 'r', encoding='utf-8') as infile:
        data_all = [json.loads(line.strip()) for line in infile]
    with open(f'./output/mscoco_{type}_-100_flitered.jsonl', 'r', encoding='utf-8') as infile:
        data_flitered = [json.loads(line.strip()) for line in infile]

    fliterer_img = [item["image_file"] for item in data_flitered]
    data_insection = []
    print(len(data_all))
    print(len(data_flitered))

    for item in tqdm(data_all):
        if item["image_file"] not in fliterer_img:
            data_insection.append(item)

    print(len(data_insection))
    with open(f'./output/mscoco_{type}_-100_insection.jsonl', 'w', encoding='utf-8') as outfile:
        for item in data_insection:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

find_insection('background_change')