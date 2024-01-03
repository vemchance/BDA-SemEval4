#!/usr/bin/env python3
import os
import sys
import io
import json

import tensorflow as tf

from lib.handle_open import handle_open
from lib.glove.glove import GloVe

### ENV VARS ################################################################
if not "INPUT" in os.environ:
	raise Exception("Error: INPUT environment variable not specified.")
if not "OUTPUT" in os.environ:
	raise Exception("Error: OUTPUT environment variable not specified.")
if not "GLOVE" in os.environ:
	raise Exception("Error: GLOVE environment variable not specified.")

INPUT=os.environ["INPUT"]
OUTPUT=os.environ["OUTPUT"]
GLOVE=os.environ["GLOVE"]

if not os.path.exists(INPUT):
	raise Exception(f"Error: INPUT filepath {INPUT} does not exist.")
if not os.path.exists(GLOVE):
	raise Exception(f"Error: GLOVE filepath {GLOVE} does not exist.")




### EMBEDDING ################################################################


glove = GloVe(GLOVE)


handle_out = handle_open(OUTPUT, "w")
with handle_open(INPUT, "r") as handle_in:
	i=0
	for line in handle_in:
		line = line.strip()
		if line == "":
			continue
		
		try:
			line = line.decode('utf-8')[:76]
		except UnicodeDecodeError as error:
			sys.stderr.write("\n"+repr(error))
			continue
		
		try:
			embedded = glove.embeddings(line)[0]
		except Exception as error:
			sys.stderr.write("\n"+repr(error))
			continue
		
		handle_out.write((json.dumps(embedded) + "\n").encode("utf-8"))
		
		if i % 100 == 0:
			sys.stderr.write(f"Processed {i} words\r")
		
		i += 1

handle_out.close()
