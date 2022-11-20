import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import pdb
import copy
from random import choice
from bert.tokenization_bert import BertTokenizer
from textblob import TextBlob


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
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.input_ids1 = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        self.chuqu = ['left', 'right', 'in', 'on', 'with', 'bottom', 'red', 'blue', \
                      'black', 'yellow', 'white', 'green', 'black', 'first', 'second', 'i']
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            sentences_for_ref1 = []
            attentions_for_ref = []
            # print(11111111111111111111111)
            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                padded_input_ids1 = [0] * self.max_tokens
                # print(sentence_raw)

                blob = TextBlob(sentence_raw.lower())
                chara_list = blob.tags
                mask_ops = []
                mask_ops1 = []
                for word_i, (word_now, chara) in enumerate(chara_list):
                    if (chara == 'NN' or chara == 'NNS') and word_i < 19 and word_now.lower() not in self.chuqu:
                        mask_ops.append(word_i)
                        mask_ops1.append(word_now)
                # print(mask_ops1 + [sentence_raw])


                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # print([sentence_raw] + input_ids)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)
                if len(mask_ops) == 0:
                    attention_remask = attention_mask
                    input_ids1 = input_ids
                else:
                    could_mask = choice(mask_ops)
                    attention_remask = copy.deepcopy(attention_mask)
                    # print([sentence_raw], mask_ops1, [could_mask], attention_mask)
                    attention_remask[could_mask + 1] = 0
                    input_ids1 = copy.deepcopy(input_ids)
                    input_ids1[could_mask + 1] = 0
                    # print(attention_remask, input_ids1)
                padded_input_ids1[:len(input_ids1)] = input_ids1
                # print(attention_remask, attention_mask)
                # print(input_ids1, input_ids)
                # pdb.set_trace()

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                sentences_for_ref1.append(torch.tensor(padded_input_ids1).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.input_ids1.append(sentences_for_ref1)
            self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        # print(this_ref_id)
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

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
            # print(1111111111111)
            embedding = []
            embedding1 = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                e1 = self.input_ids1[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                embedding1.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            tensor_embeddings1 = torch.cat(embedding1, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            tensor_embeddings1 = self.input_ids1[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            if hflip and ((2157 in tensor_embeddings) or 2187 in tensor_embeddings):
                if 2157 in tensor_embeddings:
                    # print(11111111111111)
                    tensor_embeddings = 2187 * (tensor_embeddings == 2157) + tensor_embeddings * (tensor_embeddings != 2157)
                    tensor_embeddings1 = 2187 * (tensor_embeddings1 == 2157) + tensor_embeddings1 * (tensor_embeddings1 != 2157)
                elif 2187 in tensor_embeddings:
                    tensor_embeddings = 2157 * (tensor_embeddings == 2187) + tensor_embeddings * (tensor_embeddings != 2187)
                    tensor_embeddings1 = 2157 * (tensor_embeddings1 == 2187) + tensor_embeddings1 * (tensor_embeddings1 != 2187)
                    # print(bb, tensor_embeddings)
            # print(tensor_embeddings.shape)

        return img, target, tensor_embeddings, tensor_embeddings1, attention_mask
