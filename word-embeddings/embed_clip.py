#!/usr/bin/env python3
import os
import sys
import io
import json

import torch
import clip

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




### EMBEDDING ################################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


handle_out = handle_open(OUTPUT, "w")
with handle_open(INPUT, "r") as handle_in:
	i=0
	for line in handle_in:
		line = line.strip()
		if line == "":
			continue
		
		# print("DEBUG:line", line.decode('utf-8'))
		
		with torch.no_grad():
			try:
				text = clip.tokenize([line.decode('utf-8')[:76]]).to(device)
				embedded = model.encode_text(text).tolist()[0]
			except RuntimeError as error:
				sys.stderr.write(repr(error))
				continue
			
		handle_out.write((json.dumps(embedded) + "\n").encode("utf-8"))
		
		if i % 100 == 0:
			sys.stderr.write(f"Processed {i} words\r")
		
		i += 1

handle_out.close()
