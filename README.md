# BioSEPBERT
This repository provides the code for BioSEPBERT, a neuroscience representation model designed for Brain Region text mining tasks such as named entity recognition and relation extraction.

# Install
You can use requirements.txt to install BioSEPBERT as follows (Python version >= 3.8):  
```  
pip install -r requirements.txt  
```
If you want to install the cuda version, do the following:
```  
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# Model Download
We provide two versions of pre-trained weights.  
- [BioSEPBERT-NER](https://drive.google.com/drive/folders/1qdpBYKRhDZM4z9xgBwC63SfI_RCfLam6?usp=sharing) - fine-tuning on WhiteText corpus  
- [BioSEPBERT-RE](https://drive.google.com/drive/folders/1ISGetHt3Ln-dSgjgEdoBu3XeTXQuCj28?usp=sharing) - fine-tuning on WhiteText connectivity corpus

You can also use other pre-trained weights as follows:  
- [BioBERT](https://drive.google.com/drive/folders/1YQ081Q0Z7qrEcFsove7iflDV0W4byyrU?usp=sharing) - fine-tuning on biomedical corpus
- [PubMedBERT](https://drive.google.com/drive/folders/1IjaxywOOyeocPDiBlaeScwgxYFDh5lH1?usp=drive_link) - fine-tuning on biomedical corpus

# Datasets
We provide a pre-processed version of benchmark datasets for each task as follows:  
[Named Entity Recognition](https://github.com/Brainsmatics/BioSEPBERT/tree/main/dataset/NER): (36.3 MB), a dataset on brain region named entity recognition  
[Relation Extraction](https://github.com/Brainsmatics/BioSEPBERT/tree/main/dataset/RE): (46.6 MB), 2 datasets on brain region connectivity relation extraction  
You can get all these on the [dataset](https://drive.google.com/drive/folders/1XHLfWZYgn7mu-Dmo8BaFzDz7coem8P4N?usp=sharing) folder.

# Fine-tuning
After downloading one of the pre-trained weights, unpack it to `/model`.

## Named Entity Recognition (NER)
Following command runs fine-tuning code on NER with default arguments.  
```  
python run_ner.py --task_name=BioSEPBERT --data_dir=../dataset/NER/1 --model_dir=../model/ --model_name=BioSEPBERT --model_type=BioSEPBERT --output_dir=../ --max_length=512 --train_batch_size=16 --eval_batch_size=16 --learning_rate=5e-5 --epochs=3 --logging_steps=-1 --save_steps=10 --seed=2022 --do_train --do_predict
```

## Relation Extraction (RE)
Following command runs fine-tuning code on RE with default arguments.  
```  
python run_re.py --task_name=BioSEPBERT --data_dir=../dataset/RE/1 --model_dir=../model/ --model_name=BioSEPBERT --model_type=BioSEPBERT --output_dir=../ --max_length=512 --train_batch_size=16 --eval_batch_size=16 --learning_rate=5e-5 --epochs=3 --warmup_proportion=0.1 --earlystop_patience=100 --max_grad_norm=0.0 --logging_steps=-1 --save_steps=1 --seed=2021 --do_train --do_predict
```
