#!/usr/bin/env python3
import os
import sys
import io
import json

from transformers import BertTokenizer, BertModel
import torch

from lib.handle_open import handle_open


### ENV VARS ################################################################
if not "INPUT" in os.environ:
	raise Exception("Error: INPUT environment variable not specified.")
if not "OUTPUT" in os.environ:
	raise Exception("Error: OUTPUT environment variable not specified.")

INPUT=os.environ["INPUT"]
OUTPUT=os.environ["OUTPUT"]

if not os.path.exists(INPUT):
	raise Exception(f"Error: INPUT filepath {INPUT} does not exist.")




### FUNCTIONS ################################################################

def flatten(xss):
	"""
	Ref https://stackoverflow.com/a/952952/1460422
	"""
	return [x for xs in xss for x in xs]


### EMBEDDING ################################################################


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', device=device)
model = BertModel.from_pretrained('bert-base-multilingual-uncased', return_dict=True).to(device)



handle_out = handle_open(OUTPUT, "w")
with handle_open(INPUT, "r") as handle_in:
	i=0
	for line in handle_in:
		line = line.strip()
		if line == "":
			continue
		
		try:
			if type(line) is bytes:
				line = line.decode("utf-8")
		except UnicodeDecodeError as error:
			sys.stderr.write("\n"+repr(error))
			continue
		
		
		try:
			# Ref https://huggingface.co/transformers/v3.5.1/model_doc/bert.html#transformers.BertModel.forward
			inputs = tokenizer(line, return_tensors="pt").to(device)
			outputs = model(**inputs)

			last_hidden_states = outputs.last_hidden_state.flatten()
		except Exception as error:
			sys.stderr.write("\n"+repr(error))
			continue		
		
		
		handle_out.write((line + "\t" + "\t".join([str(value) for value in last_hidden_states]) + "\n").encode("utf-8"))
		
		if i % 100 == 0:
			sys.stderr.write(f"Processed {i} words\r")
		
		i += 1

handle_out.close()
