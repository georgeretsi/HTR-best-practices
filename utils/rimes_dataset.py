import io, os
import numpy as np 
from skimage import io as img_io
from utils.word_dataset import WordLineDataset, LineListIO
from os.path import isfile
from utils.auxilary_functions import image_resize, centered
from skimage.transform import resize
import tqdm
#
from PIL import Image
import xml.etree.ElementTree as ET


class RimesDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, transforms, character_classes):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms, character_classes)
        self.setname = 'RIMES_ICDAR2011'
        self.training_lines = 'training_2011.xml'
        self.test_lines = 'eval_2011_annotated.xml'
        if segmentation_level == 'line':
            self.root_dir = '{}/{}/linelevel'.format(basefolder, self.setname)
            if subset == 'train':
                self.xmlfile = '{}/{}'.format(self.root_dir, self.training_lines)
            elif subset == 'test':
                self.xmlfile = '{}/{}'.format(self.root_dir, self.test_lines)
            else:
                raise ValueError('partition must be one of None, train or test')
        elif segmentation_level == 'word':
            self.root_dir = '{}/{}/wordlevel'.format(basefolder, self.setname)
            if subset == 'train':
                self.gtfile = 'groundtruth_training_icdar2011.txt'
                self.data_subfolder = 'training_WR'
            elif subset == 'val':
                self.gtfile = 'ground_truth_validation_icdar2011.txt'
                self.data_subfolder = 'testdataset_ICDAR'
            elif subset == 'test':
                self.gtfile = 'grount_truth_test_icdar2011.txt'
                self.data_subfolder = 'data_test'
        else:
            raise ValueError('Segmentation level must be either word or line level.')
        self.data = [] # A list of tuples (image data, transcription)
        self.query_list = None
        self.dictionary_file = 'test_dictionary_task_wr_icdar_2011.txt' #Unused in this version
        super().__finalize__()

    def main_loader(self, partition, level) -> list:
        ##########################################
        # Load pairs of (image, ground truth)
        ##########################################
        # load the dataset
        def relu(x):
            if(x < 0):
                return(0)
            return(x)
        data = []
        if(level == 'word'):
            words_parsed = 0
            words_failed_to_open = 0
            with io.open(os.path.join(self.root_dir, self.gtfile), mode='r', encoding='utf-8-sig') as fp:                
                annotation_lines = fp.readlines()
            for line in tqdm.tqdm(annotation_lines):
                try:
                    imagefile, text = line.strip().split(' ')
                except:
                    continue
                try:
                    word_img = img_io.imread(os.path.join(self.root_dir, self.data_subfolder, imagefile))
                except:
                    words_failed_to_open += 1
                    continue
                word_img = 1 - word_img.astype(np.float32) / 255.0
                data.append(
                    (word_img, text)
                )
                self.print_random_sample(word_img, text, words_parsed, as_saved_files=False)
                words_parsed += 1
            print('Read {} words succesfully. {} words failed to load.'.format(words_parsed, words_failed_to_open))
        elif(level == 'line'):
            root = ET.parse(self.xmlfile).getroot()
            lines_parsed = 0
            for singlepage in tqdm.tqdm(root.findall('SinglePage')):
                doc_img = img_io.imread(os.path.join(self.root_dir, 'images_gray', os.path.basename(singlepage.get('FileName'))))
                doc_img = 1 - doc_img.astype(np.float32) / 255.0            
                for paragraph in singlepage:
                    for line in paragraph:
                        text = line.get('Value')#.decode('UTF-8')
                        bottom = line.get('Bottom')
                        left = line.get('Left')
                        right = line.get('Right')
                        top = line.get('Top')
                        #
                        top, bottom, left, right = relu(int(top)), relu(int(bottom)), relu(int(left)), relu(int(right))
                        lines_parsed += 1
                        token_img = doc_img[int(top):int(bottom), int(left):int(right)].copy()
                        #token_img = self.check_size(img=token_img, min_image_width_height=self.fixed_size[0])
                        token_img = image_resize(token_img, height=token_img.shape[0] // 2)
                        #print(token_img.shape)
                        data.append(
                            (token_img, text)
                        )
                        self.print_random_sample(token_img, text, lines_parsed, as_saved_files=False)
            print('For partition {}, {} {} tokens have been parsed'.format(partition, lines_parsed, level))
        else:
            raise ValueError('Segmentation level must be either line or word.')
        return(data)