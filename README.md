# HTR-best-practices

Basic HTR concepts/modules to boost performance. Official code for the paper ["Best Practices for a Handwritten Text Recognition system"](https://arxiv.org/abs/2404.11339) - presented in 15th IAPR International Workshop on Document Analysis Systems (DAS 2022).


## Updates 

05/06/2024: 

The whole code was reworked towards a more clear framework, that can be easily modified and adapted with new architectures.

Update Highlights:
- restructured code: configuration file is .yaml for high level configuration of the hyper-parameters.
- Albumentation package is used for augmentation.
- using NLTK-based tokenizer for WER. Different groundtruth has different annotation w.r.t. spaces that can change significantly the overall WER metric. For example ascii gt files of IAM have spaces between words and commas, while the original xml gt do not. 
- better performance than the paper, namely 4.2% CER in the IAM line-level scenario.



## Installation

You need to have a working version of PyTorch installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:

```bash
conda create -n htr python=3.9
conda activate htr
pip install -r requirements.txt
```

## Data Preparation

This repo contains all the required steps for training and evaluatin on [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). To access the data, you can register [here](https://fki.tic.heia-fr.ch/register).

For the line-level setup, which is currently supported by this repo, you only need to download the form images (3sets: data/formsA-D.tgz, data/formsE-H.tgz, data/formsI-Z.tgz - unzip them into a common folder of images) and the xml groundtruth (data/xml.tgz). All these files can be found in the [official website](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) after registration.

Then, we can create a line-level instatiation of the dataset through the script:
```bash
python prepare_iam.py $mypath$/IAM/forms/ $mypath$/IAM/xml/ ./data/IAM/splits/ ./data/IAM/processed_lines
```

where $mypath$/IAM is the path where the different IAM files are saved. The splits are provided in the local path ./data. Finally, the last argument is the output folder.

Note that IAM provides already segmented lines, but the provided images have masked-out the background at word-level - while there are some lines missing w.r.t. the xml files. To have a more realistic setup, we extract from the initial forms, the requested lines.

## Training

The training is performed as follows:
```bash
python trainer.py config.yaml
```

We can define the gpu to run: 
```bash
CUDA_VISIBLE_DEVICES=0 python trainer.py config.yaml
```
or 
```bash
python trainer.py config.yaml device='cuda:0'
```

We can change the overall setup directly through config.yaml or as extra arguments to the main python command, as this example:
```bash
python trainer.py config.yaml train.lr=1e-3 arch.head_type='both' train.num_epochs=800
```

One of the main elements of this paper, was the introduction of a shortcut head. This is selected in the above command with the head_type='both' option.


## Testing

A pre-trained model is provided in saved_models path (htrnet.pt). You can use it, or a re-trained one, to evaluate IAM dataset:
```bash
python evaluate.py config.yaml resume=./saved_models//htrnet.pt 
```

Also a single image demo version is available, where an image is selected from the test set:
```bash
python demo.py config.yaml resume=./saved_models/htrnet.pt ./data/IAM/processed_lines/test/c04-165-05.png
```

## Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{retsinas2022best,
  title={Best practices for a handwritten text recognition system},
  author={Retsinas, George and Sfikas, Giorgos and Gatos, Basilis and Nikou, Christophoros},
  booktitle={International Workshop on Document Analysis Systems},
  pages={247--259},
  year={2022},
  organization={Springer}
}
```

