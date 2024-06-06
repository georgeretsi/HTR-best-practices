import io,os
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from utils.preprocessing import load_image, preprocess

class HTRDataset(Dataset): 
    def __init__(self, 
        basefolder: str = 'IAM/',                #Root folder
        subset: str = 'train',                          #Name of dataset subset to be loaded. (e.g. ''train', 'val', 'test')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        ):
        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes

        # load gt.txt from basefolder - each line contains a path to an image and its transcription
        data = []
        with open(os.path.join(basefolder, subset, 'gt.txt'), 'r') as f:
            for line in f:
                img_path, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])
                data += [(os.path.join(basefolder, subset, img_path + '.png'), transcr)]

        self.data = data

        if self.character_classes is None:
            res = set()
            for _,transcr in data:
                res.update(list(transcr))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res 

    def __getitem__(self, index):
        img_path = self.data[index][0]
        transcr = " " + self.data[index][1] + " "
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        img = load_image(img_path)

        if self.subset == 'train':
            nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
            nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
            img = resize(image=img, output_shape=(nheight, nwidth)).astype(np.float32)

        img = preprocess(img, (fheight, fwidth))

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = torch.Tensor(img).float().unsqueeze(0)
        return img, transcr
    
    def __len__(self):
        return len(self.data)
