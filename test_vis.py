import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision
from bert.tokenization_bert import BertTokenizer

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
import cv2
import mmcv

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes



def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

import pickle
def main(args):

    save_output_mask_dir = 'vis_output_mask_refcoco+_test'
    # save_target_dir = 'vis_target_refcoco+_val_data'
    # ref_file = 'refer/data/refcocog/refs(umd).p'

    # ref_data = pickle.load(open(ref_file, 'rb'))

    if not os.path.exists(save_output_mask_dir):
        import pdb
        pdb.set_trace()
        os.mkdir(save_output_mask_dir)
        
    device = torch.device(args.device)
    

    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    model.eval()
    with torch.no_grad():
        data_dir = 'refer/data/images/mscoco/images/train2014/'
        filename = 'COCO_train2014_000000380440.jpg'
        # image = mmcv.imread(os.path.join(data_dir,filename))
        image = Image.open(os.path.join(data_dir,filename)).convert("RGB")
        save_img = np.array(image)
        transform = get_transform(args)

        image,img1 = transform(image,image)
        image = torch.cat((image,image),dim=2)

        sentences= ['dog','white','right','person','bike']
        for sentence_raw in sentences:
            max_tokens = 20
            tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
            sentences = tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            attentions = [0] * max_tokens
            padded_sentences = [0] * max_tokens
            # truncation of tokens
            sentences = sentences[:max_tokens]

            padded_sentences[:len(sentences)] = sentences
            attentions[:len(sentences)] = [1]*len(sentences)


            
            padded_sentences = torch.tensor(padded_sentences).unsqueeze(0)
            attentions = torch.tensor(attentions).unsqueeze(0)
            image, padded_sentences, attentions = image.to(device),padded_sentences.to(device), attentions.to(device)
            padded_sentences = padded_sentences.squeeze(1)
            attentions = attentions.squeeze(1)


            if bert_model is not None:
                last_hidden_states = bert_model(padded_sentences, attention_mask=attentions)[0]

                embedding = last_hidden_states.permute(0, 2, 1)
                output = model(image.unsqueeze(0), embedding, attentions.unsqueeze(-1))
            else:
                output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

            output = output.cpu()
            output_mask = output.argmax(1).data.numpy()
            import pdb
            pdb.set_trace()
            cv2.imwrite(os.path.join(save_output_mask_dir, filename.replace('.jpg','_') +'{}.png'.format(sentence_raw)),output_mask[0]*255)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
