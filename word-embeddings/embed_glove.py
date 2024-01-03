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




### FUNCTIONS ################################################################

def flatten(xss):
	"""
	Ref https://stackoverflow.com/a/952952/1460422
	"""
	return [x for xs in xss for x in xs]


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
			if type(line) is bytes:
				line = line.decode("utf-8")
		except UnicodeDecodeError as error:
			sys.stderr.write("\n"+repr(error))
			continue
		
		
		try:
			embedded = glove.embeddings(line)
		except Exception as error:
			sys.stderr.write("\n"+repr(error))
			continue
		if len(embedded) == 0:
			sys.stderr.write(f"\nError: Failed to embed with GloVe, ignoring.\n")
			continue
		embedded = embedded[0]
		
		handle_out.write((line + "\t" + "\t".join([str(word) for word in embedded]) + "\n").encode("utf-8"))
		
		if i % 100 == 0:
			sys.stderr.write(f"Processed {i} words\r")
		
		i += 1

handle_out.close()
