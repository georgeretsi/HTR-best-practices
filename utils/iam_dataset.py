import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize, centered

class IAMDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = 'IAM'
        self.trainset_file = '{}/{}/set_split/trainset.txt'.format(self.basefolder, self.setname)
        self.valset_file = '{}/{}/set_split/validationset1.txt'.format(self.basefolder, self.setname)
        self.testset_file = '{}/{}/set_split/testset.txt'.format(self.basefolder, self.setname)
        self.line_file = '{}/{}/ascii/lines.txt'.format(self.basefolder, self.setname)
        self.word_file = '{}/{}/ascii/words.txt'.format(self.basefolder, self.setname)
        self.word_path = '{}/{}/words'.format(self.basefolder, self.setname)
        self.line_path = '{}/{}/lines'.format(self.basefolder, self.setname)
        #self.stopwords_path = '{}/{}/iam-stopwords'.format(self.basefolder, self.setname)
        super().__finalize__()

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_iam_info(self, set='train', level='word'):
            if subset == 'train':
                #valid_set = np.loadtxt(self.trainset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/train.uttlist', dtype=str)
                #print(valid_set)
            elif subset == 'val':
                #valid_set = np.loadtxt(self.valset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/validation.uttlist', dtype=str)
            elif subset == 'test':
                #valid_set = np.loadtxt(self.testset_file, dtype=str)
                valid_set = np.loadtxt('./utils/aachen_iam_split/test.uttlist', dtype=str)
            else:
                raise ValueError
            if level == 'word':
                gtfile= self.word_file
                root_path = self.word_path
            elif level == 'line':
                gtfile = self.line_file
                root_path = self.line_path
            else:
                raise ValueError
            gt = []
            for line in open(gtfile):
                if not line.startswith("#"):
                    info = line.strip().split()
                    name = info[0]
                    name_parts = name.split('-')
                    pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
                    if level == 'word':
                        line_name = pathlist[-2]
                        del pathlist[-2]
                    elif level == 'line':
                        line_name = pathlist[-1]
                    form_name = '-'.join(line_name.split('-')[:-1])
                    #if (info[1] != 'ok') or (form_name not in valid_set):
                    if (form_name not in valid_set):
                        #print(line_name)
                        continue
                    img_path = '/'.join(pathlist)
                    transcr = ' '.join(info[8:])
                    gt.append((img_path, transcr))
            return gt

        info = gather_iam_info(self, subset, segmentation_level)
        data = []
        for i, (img_path, transcr) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))
            #

            try:
                img = img_io.imread(img_path + '.png')
                img = 1 - img.astype(np.float32) / 255.0
                img = image_resize(img, height=img.shape[0] // 2)
            except:
                continue
                
            #except:
            #    print('Could not add image file {}.png'.format(img_path))
            #    continue

            # transform iam transcriptions
            transcr = transcr.replace(" ", "")
            # "We 'll" -> "We'll"
            special_cases  = ["s", "d", "ll", "m", "ve", "t", "re"]
            # lower-case 
            for cc in special_cases:
                transcr = transcr.replace("|\'" + cc, "\'" + cc)
                transcr = transcr.replace("|\'" + cc.upper(), "\'" + cc.upper())

            transcr = transcr.replace("|", " ")

            data += [(img, transcr)]
        return data