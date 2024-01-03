#!/usr/bin/env python3
import os
import sys
import gzip
import time
import json

from loguru import logger
import umap
import umap.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datashader as ds
import colorcet

from lib.io.handle_open import handle_open
from lib.glove.glove import GloVe
from lib.glove.normalise_text import normalise as normalise_text

if "--help" in sys.argv:
	print("""Wordlist → UMAP convertificator and plotificator 9000
	By Starbeamrainbowlabs

Usage:
	[ENV_VAR=value ....] path/to/umapify.py

Environment variables:
	INPUT   	The path to the wordlist file. See the command below for more info.
	OUTPUT  	The path to the output tsv file. Will have DIM+1 columns in the form '[ word, dim_1, dim_2, ... dim_x ] @ tsv'. A sister file will be placed with the file extension .png with a Cool Plot™
	DIM			The number of output dimensions to UMAP to.
""")
	exit(0)

# ███████ ███    ██ ██    ██
# ██      ████   ██ ██    ██
# █████   ██ ██  ██ ██    ██
# ██      ██  ██ ██  ██  ██
# ███████ ██   ████   ████

FILEPATH_INPUT = os.environ["INPUT"] if "INPUT" in os.environ else None
FILEPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else None
DIM = int(os.environ["DIM"]) if "DIM" in os.environ else 2

filepath_output_image = os.path.join(
	os.path.dirname(FILEPATH_OUTPUT),
	os.path.splitext(os.path.basename(
		FILEPATH_OUTPUT.replace(".gz", "")
	))[0] # The .png is added automatically by datashader
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

if FILEPATH_INPUT is None or not os.path.exists(FILEPATH_INPUT):
	raise Exception(f"Error: No file found at '{FILEPATH_INPUT}'. Either it doesn't exist, or you don't have permission to read it.")
if FILEPATH_OUTPUT is None or FILEPATH_OUTPUT == "":
	raise Exception(f"Error: No output filepath specified.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger.info("Embedded Wordlist → UMAP convertificator and plotificator 9000")
for env_name in ["FILEPATH_INPUT", "FILEPATH_OUTPUT", "filepath_output_image", "DIM"]:
	logger.info(f"> {env_name} {str(globals()[env_name])}")


# ██████   █████  ████████  █████
# ██   ██ ██   ██    ██    ██   ██
# ██   ██ ███████    ██    ███████
# ██   ██ ██   ██    ██    ██   ██
# ██████  ██   ██    ██    ██   ██

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

start = time.time()
words = []
embeds = []
with handle_open(FILEPATH_INPUT, "r") as handle:
	stop_words_skipped = 0
	words_read = 0
	for line in handle:
		if type(line) is bytes:
			line = line.decode("utf-8")
		row = line.rstrip("\n").split("\t", maxsplit=1)
		word = row[0]
		if line == "" or len(row) < 2:
			continue
		
		embed = [int(x) for x in row[1:]]
		
		words.append(word)
		embeds.append(embed)
		
		words_read += 1
		if words_read % 1000 == 0:
			sys.stderr.write(f"Reading words: {words_read} words read so far\r")
		

logger.info(f"{len(words)} read in {round(time.time() - start, 3)}s, {stop_words_skipped} stop words skipped")


# ██    ██ ███    ███  █████  ██████  
# ██    ██ ████  ████ ██   ██ ██   ██ 
# ██    ██ ██ ████ ██ ███████ ██████  
# ██    ██ ██  ██  ██ ██   ██ ██      
#  ██████  ██      ██ ██   ██ ██      

logger.info("UMAPing...")
umapped = umap.UMAP(
	min_dist=0.05,
	n_components=DIM
).fit_transform(embeds)
logger.info("UMAP conversion complete")



# dim_reducer = umap.UMAP(
# 	min_dist=0.05  # default: 0.1
# ).fit(words_glove)


# ██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████
# ██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██
# ██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███
# ██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██
# ██      ███████  ██████     ██       ██    ██ ██   ████  ██████

def plot(filepath_target, umapped, dim):
	logger.info("Plotting")
	if dim == 2:
		df = pd.DataFrame(umapped)
		df.columns = ["x", "y"]
		
		print(df)
		
		canvas = ds.Canvas(plot_width=850, plot_height=850)
		points = canvas.points(df, "x", "y")
		result = ds.tf.set_background(ds.tf.shade(points), color="white")
		ds.utils.export_image(
			result,
			filepath_target
		)
		print("canvas", canvas, "points", points, "result", result)
		logger.info(f"Written plot with 2 dimensions to {filepath_target}.png")
	else:
		logger.info(f"Warning: Not exporting a plot, since a dim of {dim} is not supported (supported values: 2).")

def save_tsv(filepath_target, umapped, words):
	logger.info("Writing tsv")
	with handle_open(filepath_target, "w") as handle:
		print(umapped[0:10])
		print(words[0:10])
		# df_points = pd.DataFrame(umapped)
		# df_labels = pd.DataFrame(words)
		# df_labels.columns = ["word"]
		
		rows = [ [ row[0], *row[1] ] for row in zip(words, umapped) ]
		
		for row in rows:
			payload = "\t".join([str(item) for item in row]) + "\n"
			handle.write(payload.encode() if filepath_target.endswith(".gz") else payload)
	
	logger.info(f"Written values to {filepath_target}")

plot(
	filepath_target=filepath_output_image,
	umapped=umapped,
	dim=DIM
)
save_tsv(FILEPATH_OUTPUT, umapped, words)