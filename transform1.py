import torchvision
from lib import segmentation
import pdb
import transforms as T


class new_transform(object):
    def __init__(self, args):
        self.Resize = T.Resize(args.img_size, args.img_size)
        self.HorizontalFlip = T.RandomHorizontalFlip1(0.5)
        self.ToTensor = T.ToTensor()
        self.Norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RandomResize = T.RandomResize(min_size=440, max_size=520)
        self.RandomCrop = T.RandomCrop(args.img_size)

    def __call__(self, image, target):
        image, target = self.Resize(image, target)
        image, target, flip_flag = self.HorizontalFlip(image, target)
        # image, target = self.RandomResize(image, target)
        # image, target = self.RandomCrop(image, target)
        image, target = self.ToTensor(image, target)
        image, target = self.Norm(image, target)

        return image, target, flip_flag


# def get_transform(args):
#     transforms = [T.Resize(args.img_size, args.img_size),
#                   T.ToTensor(),
#                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                   ]
#
#     return T.Compose(transforms)