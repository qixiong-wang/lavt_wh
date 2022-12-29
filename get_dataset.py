import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random, csv, pdb

from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        # print(1111111)
        # print(ref_ids)

        self.augref_idx_dict = {}
        self.imgid_ref_dict = {}
        for ref_id in ref_ids:
            try:
                self.imgid_ref_dict[self.refer.Refs[ref_id]['image_id']].append(ref_id)
            except:
                self.imgid_ref_dict[self.refer.Refs[ref_id]['image_id']] = [ref_id]
        # 上面那个try except，其实就是把refer id给加到image id里。一个refer_id里可能有多个sentence，这多个sentence都对应是一个物体。所以这里其实也可以指示\
        # 每个图片里有多少个target会被用于分割。
        # print(self.refer.Refs[ref_id])
        # self.imgs_ref_dict = list(all_imgs[i] for i in img_ids)

        self.ref_ids = ref_ids #这里就是一串数字
        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        max_ref_id = max(ref_ids)
        index_num = 0
        pdb.set_trace()

        for r in ref_ids:
            ref = self.refer.Refs[r]
            # print(ref)

            sentences_for_ref = []
            attentions_for_ref = []
            # pdb.set_trace()


            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']

                index_num = index_num + 1
                write_now = [index_num]
                write_now.append(ref['file_name'])
                write_now.append('or_refcoco')
                write_now.append(ref['ann_id'])
                write_now.append(ref['ref_id'])
                write_now.append(ref['image_id'])
                write_now.append(sent_id)
                write_now.append(sentence_raw)
                ff1.writerow(write_now)
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1] * len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

        print('work done')
        # pdb.set_trace()

        self.org_len = len(self.ref_ids)
        if split == 'train':
            for img_id_ref in self.imgid_ref_dict:
                if len(self.imgid_ref_dict[img_id_ref]) >= 2:
                    ref_ids = random.sample(self.imgid_ref_dict[img_id_ref], 2)
                    max_ref_id += 1
                    self.augref_idx_dict[max_ref_id] = ref_ids

                    sentences_for_ref = []
                    attentions_for_ref = []
                    for num in range(4):
                        try:
                            sentence_concat = (self.refer.Refs[ref_ids[0]]['sentences'][num // 2]['raw'] + ' and ' +
                                               self.refer.Refs[ref_ids[1]]['sentences'][num % 2]['raw'])
                            input_ids = self.tokenizer.encode(text=sentence_concat, add_special_tokens=True)
                            # print([sentence_concat], input_ids)

                            # truncation of tokens
                            input_ids = input_ids[:self.max_tokens]
                            padded_input_ids[:len(input_ids)] = input_ids
                            attention_mask[:len(input_ids)] = [1] * len(input_ids)
                            sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                            attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
                        except:
                            pass
                    self.ref_ids.append(max_ref_id)
                    self.input_ids.append(sentences_for_ref)
                    self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):

        if index < self.org_len:
            this_ref_id = self.ref_ids[index]
            ref = self.refer.loadRefs(this_ref_id)
            # print(this_ref_id, ref)
            ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1
        else:
            this_ref_id = self.augref_idx_dict[self.ref_ids[index]]
            ref_mask_0 = np.array(self.refer.getMask(self.refer.Refs[this_ref_id[0]])['mask'])
            ref_mask_1 = np.array(self.refer.getMask(self.refer.Refs[this_ref_id[1]])['mask'])
            annot = np.zeros(ref_mask_0.shape)
            annot[ref_mask_0 == 1] = 1
            annot[ref_mask_1 == 1] = 1


        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]
        # print(this_img)


        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            # img, target = self.image_transforms(img, annot)
            if self.split == 'train':
                img, target, hflip = self.image_transforms(img, annot)
            elif self.split == 'val':
                img, target = self.image_transforms(img, annot)
                hflip = False
            # print(self.input_ids)

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            if hflip and ((2157 in tensor_embeddings) or (2187 in tensor_embeddings)):
                # if 2157 in tensor_embeddings:
                #     # print(11111111111111)
                #     tensor_embeddings = 2187 * (tensor_embeddings == 2157) + tensor_embeddings * (tensor_embeddings != 2157)
                # elif 2187 in tensor_embeddings:
                #     tensor_embeddings = 2157 * (tensor_embeddings == 2187) + tensor_embeddings * (tensor_embeddings != 2187)
                #     # print(bb, tensor_embeddings)
                #     print(tensor_embeddings.shape)
                tensor_embeddings = (2187 * (tensor_embeddings == 2157) + 2157 * (tensor_embeddings == 2187)) + \
                                    (tensor_embeddings * (tensor_embeddings != 2157) * (tensor_embeddings != 2187))
                # print(bb, tensor_embeddings)
            # print(bb, tensor_embeddings)


        return img, target, tensor_embeddings, attention_mask




if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    ff = open('./linshi/ori_refcoco_testB.csv', 'w')
    ff1 = csv.writer(ff)
    ff1.writerow(['index', 'filename', 'type', 'ann_id', 'ref_id', 'image_id', 'sent_id', 'sentence_raw'])
    ds = ReferDataset(args,
                      split='testB')