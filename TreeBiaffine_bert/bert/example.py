import torch
#from transformers import BertTokenizer
from transformers import BertModel, BertForMaskedLM
from data.Dataloader import *
from driver.BertTokenHelper import BertTokenHelper
from driver.BertModel import BertExtractor
from driver.Config import *


# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('./')
## Tokenize input
# tokenized_text = tokenizer.tokenize(text)

tokenizer = BertTokenHelper('./bert')

text = "Who was Jim Henson ? Jim Henson was a puppeteer"
text1 = 'Although the plan received unanimous approval , this does not mean that it represents the lowest common denominator .'
# Mask a token that we will try to predict back with `BertForMaskedLM`
bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask = batch_pretrain_variable([text.split(), text1.split()], tokenizer)
# Load pre-trained model (weights)
config = Configurable('config.cfg', [])
bert = BertExtractor(config)
outputs = bert(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask)
# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
print(outputs)

