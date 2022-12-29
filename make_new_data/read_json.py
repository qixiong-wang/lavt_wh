import json
import pdb
import numpy as np

json_path = '/data/huyutao/coco2014/annotations/instances_train2014.json'
json_labels = json.load(open(json_path, "r"))
pdb.set_trace()
print(json_labels["info"])
cla_pools = json_labels["categories"]
coco2014_cla_ids = {}
for i in range(len(cla_pools)):
    cla0 = cla_pools[i]
    id0 = cla0['id']
    name = cla0['name']
    coco2014_cla_ids[id0] = name
    np.save('/data/huyutao/new_coco/coco2014_cla_ids.npy', coco2014_cla_ids)
    # dict = np.load('/data/huyutao/new_coco/coco2014_cla_ids.npy', allow_pickle=True).item()
pdb.set_trace()