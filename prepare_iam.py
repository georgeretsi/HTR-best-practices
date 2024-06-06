import os 
import sys
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


# main function - read the xml files and prepare the data
def prepare_iam_data(form_path, xmls_path, splits_path, output_path, pad_size=16, scale=1.0):

    # check if output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

        # check for train / val / test subdirectories
        os.makedirs(os.path.join(output_path, "train"))
        os.makedirs(os.path.join(output_path, "val"))
        os.makedirs(os.path.join(output_path, "test"))

    # check if form directory exists
    if not os.path.exists(form_path):
        print("Form directory does not exist")
        return None
    
    # check if xml directory exists
    if not os.path.exists(xmls_path):
        print("XML directory does not exist")
        return None

    # check if splits directory exists
    if not os.path.exists(splits_path):
        print("Splits directory does not exist")
        return None
    
    # get the list of xml files
    xml_files = os.listdir(xmls_path)

    train_set = np.loadtxt(os.path.join(splits_path, 'train.uttlist'), dtype=str)
    val_set = np.loadtxt(os.path.join(splits_path, 'validation.uttlist'), dtype=str)
    test_set = np.loadtxt(os.path.join(splits_path, 'test.uttlist'), dtype=str)

    gt_lines = {'train': [], 'val': [], 'test': []}
    # iterate over the xml files
    for xml_file in tqdm(xml_files):

        # get the file name
        file_name = xml_file.split(".")[0]

        if file_name in train_set:
            subset = "train"
        elif file_name in val_set:
            subset = "val"
        elif file_name in test_set:
            subset = "test"
        else:
            continue

        # get the form file
        form_file = os.path.join(form_path, file_name + ".png")

        # read the form image with PIL
        form_img = Image.open(form_file)
            
        # resize to further compress it
        form_img = form_img.resize((int(form_img.width * scale), int(form_img.height * scale))) #, Image.LANCZOS)
    
    
        # get the xml file
        xml_file = os.path.join(xmls_path, xml_file)

        # use xml parser to read the xml file
        xml_tree = ET.parse(xml_file)
        #h, w = form_img.shape

        w, h = form_img.size

        # find the <handwritten-part> tag
        handwritten_part = xml_tree.find("handwritten-part")

        # find tags starting with <line ...>
        lines = handwritten_part.findall("line")

        # for each line tag find id, text and bounding box
        for line in lines:

            # get the line id
            line_id = line.get("id")

            # get the line text
            line_text = line.get("text")
            line_text = line_text.replace("&amp;", "&")
            line_text = line_text.replace("&quot;", "\"")
            line_text = line_text.replace("&apos;", "\'")

            gt_lines[subset].append(line_id + " " + line_text + "\n")
            
            words = line.findall("word")
            hl, hu, wl, wh = 100000, 0, 100000, 0
            mask = .5 + np.zeros((h, w))
            for word in words:

                # find tag starting with <cmp ...>
                cmps = word.findall("cmp")
                for cmp in cmps:
                    # get the word bounding box
                    tx, ty, tw, th = cmp.get("x"), cmp.get("y"), cmp.get("width"), cmp.get("height")
                    tx, ty, tw, th = int(int(tx) * scale), int(int(ty) * scale), int(int(tw) * scale), int(int(th) * scale)
                    
                    mask[ty:ty+th, tx:tx+tw] = 1        

                    hl = min(int(ty), int(hl))
                    hu = max(int(ty) + int(th), int(hu))
                    wl = min(int(tx), int(wl))
                    wh = max(int(tx) + int(tw), int(wh))

            # pad_size pixels pad on top and bottom
            hl = max(0, int(hl) - pad_size)
            hu = min(int(hu) + pad_size, int(h))

            # 2 * pad_size pixels pad on left and right
            wl = max(0, int(wl) - 2 * pad_size)
            wh = min(int(wh) + 2 * pad_size, int(w))
        
            line_img = form_img.crop((int(wl), int(hl), int(wh), int(hu)))

            line_file = os.path.join(output_path, subset, line_id + ".png")

            # use PIL to save the image and compress it
            line_img.save(line_file, optimize=True, quality=60)

    # write the gt file
    for subset in gt_lines.keys():
        with open(os.path.join(output_path, subset, "gt.txt"), "w") as f:
            f.writelines(gt_lines[subset])

    return None




# main call - arguments are the paths to the form and xml directories
if __name__ == '__main__':

    # 1st argument is the path to the form directory
    form_path = sys.argv[1]

    # 2nd argument is the path to the xml directory
    xmls_path = sys.argv[2]

    # 3rd argument is the path to the splits directory
    splits_path = sys.argv[3]

    # 4rth argument is the path to the output directory
    output_path = sys.argv[4]

    # prepare the data
    prepare_iam_data(form_path, xmls_path, splits_path, output_path, pad_size=2, scale=0.5)





