# BioSEPBERT
This repository provides the code for BioSEPBERT, a neuroscience representation model designed for Brain Region text mining tasks such as named entity recognition and relation extraction.
# Download
We provide two versions of pre-trained weights.
- [BioSEPBERT-NER](http://brainsmatics.org/) - fine-tuning on WhiteText corpus
- [BioSEPBERT-RE](http://brainsmatics.org/) - fine-tuning on WhiteText connectivity corpus
# Install
After downloading the pre-trained weights, use requirements.txt to install BioSEPBERT as follows (python version >= 3.7):  
```  
pip install -r requirements.txt  
```
# Datasets
We provide a pre-processed version of benchmark datasets for each task as follows:  
[Named Entity Recognition](https://github.com/Brainsmatics/BioSEPBERT/tree/main/dataset/NER): (36.3 MB), a dataset on brain region named entity recognition  
[Relation Extraction](https://github.com/Brainsmatics/BioSEPBERT/tree/main/dataset/RE): (118.6 MB), 2 datasets on brain region connectivity relation extraction  
You can simply get all these on the [dataset](https://github.com/Brainsmatics/BioSEPBERT/tree/main/dataset) folder.
# Fine-tuning
After downloading one of the pre-trained weights, unpack it to `/model`.
## Named Entity Recognition (NER)
Following command runs fine-tuning code on NER with default arguments.  
```python run_ner.py --task_name=BioSEPBERT --data_dir=../dataset/NER/1 --model_dir=../model/ --model_name=BioSEPBERT --model_type=BioSEPBERT --output_dir=../ --max_length=512 --train_batch_size=16 --eval_batch_size=16 --learning_rate=5e-5 --epochs=3 --logging_steps=-1 --save_steps=10 --seed=2022 --do_train --do_predict```
## Relation Extraction (RE)
Following command runs fine-tuning code on RE with default arguments.  
```python run_re.py --task_name=BioSEPBERT --data_dir=../dataset/RE/1 --model_dir=../model/ --model_name=BioSEPBERT --model_type=BioSEPBERT --output_dir=../ --max_length=512 --train_batch_size=16 --eval_batch_size=16 --learning_rate=5e-5 --epochs=3 --warmup_proportion=0.1 --earlystop_patience=100 --max_grad_norm=0.0 --logging_steps=-1 --save_steps=1 --seed=2021 --do_train --do_predict```
