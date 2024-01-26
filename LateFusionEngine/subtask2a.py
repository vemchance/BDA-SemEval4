#!/usr/bin/env python3

import os
import sys
import json
import copy
import pprint

from lib.handle_open import handle_open

pp = pprint.PrettyPrinter(indent=4)


### ENV VARS ################################################################


WEIGHTS_IN=os.environ["WEIGHTS_IN"] if "WEIGHTS_IN" in os.environ else None
WEIGHTS_OUT=os.environ["WEIGHTS_OUT"] if "WEIGHTS_OUT" in os.environ else None

if not "INPUT_LABELS" in os.environ and WEIGHTS_IN is None:
	raise Exception("Error: INPUT_LABELS environment variable not specified. Either INPUT_LABELS or WEIGHTS_IN must be specified.")
if not "INPUT_NLP" in os.environ:
	raise Exception("Error: INPUT_NLP environment variable not specified.")
if not "INPUT_VISION" in os.environ:
	raise Exception("Error: INPUT_VISION environment variable not specified.")
if not "OUTPUT" in os.environ:
	raise Exception("Error: OUTPUT environment variable not specified.")


INPUT_LABELS=os.environ["INPUT_LABELS"] if "INPUT_LABELS" in os.environ else None
INPUT_NLP=os.environ["INPUT_NLP"]
INPUT_VISION=os.environ["INPUT_VISION"]
OUTPUT=os.environ["OUTPUT"]

if WEIGHTS_IN is None and WEIGHTS_OUT is None:
	raise Exception("Error: Either WEIGHTS_IN (read precalculated weights from file) or WEIGHTS_OUT (calculate weights from INPUT_NLP and INPUT_VISION, then write them to WEIGHTS_OUT) must be specified.")
if WEIGHTS_IN is not None and WEIGHTS_OUT is not None:
	raise Exception("Error: WEIGHTS_IN and WEIGHTS_OUT are mutually exclusive.")

if INPUT_LABELS is not None and not os.path.exists(INPUT_LABELS):
	raise Exception(f"Error: INPUT_LABELS filepath {INPUT_LABELS} does not exist.")
if not os.path.exists(INPUT_NLP):
	raise Exception(f"Error: INPUT_NLP filepath {INPUT_NLP} does not exist.")
if not os.path.exists(INPUT_VISION):
	raise Exception(f"Error: INPUT_VISION filepath {INPUT_VISION} does not exist.")
if WEIGHTS_IN is not None and not os.path.exists(WEIGHTS_IN):
	raise Exception(f"Error: WEIGHTS_IN filepath {WEIGHTS_IN} does not exist.")



### FUNCTIONS ################################################################

def hashmapificate(dataset):
	result = {}
	# i = 1
	for item in dataset:
		result[item["id"]] = item
		# if i >= 3:
		# 	break
		# i += 1
	
	return result

def make_empty_probabilities(val=0):
	return {
		"Appeal to (Strong) Emotions": copy.copy(val),
		"Appeal to authority": copy.copy(val),
		"Appeal to fear/prejudice": copy.copy(val),
		"Bandwagon": copy.copy(val),
		"Black-and-white Fallacy/Dictatorship": copy.copy(val),
		"Causal Oversimplification": copy.copy(val),
		"Doubt": copy.copy(val),
		"Exaggeration/Minimisation": copy.copy(val),
		"Flag-waving": copy.copy(val),
		"Glittering generalities (Virtue)": copy.copy(val),
		"Loaded Language": copy.copy(val),
		"Misrepresentation of Someone's Position (Straw Man)": copy.copy(val),
		"Name calling/Labeling": copy.copy(val),
		"Obfuscation, Intentional vagueness, Confusion": copy.copy(val),
		"Presenting Irrelevant Data (Red Herring)": copy.copy(val),
		"Reductio ad hitlerum": copy.copy(val),
		"Repetition": copy.copy(val),
		"Slogans": copy.copy(val),
		"Smears": copy.copy(val),
		"Thought-terminating cliché": copy.copy(val),
		"Transfer": copy.copy(val),
		"Whataboutism": copy.copy(val),
	}

def list2probabilities(arr):
	result = make_empty_probabilities()
	for item in arr:
		result[item] = 1
	return result

def calc_accuracies(dataset, field_target):
	result = make_empty_probabilities([0, 0]) # [true, false] wrt correct or not ref label
	
	threshold = 0.5
	
	for item_id in dataset:
		item = dataset[item_id]
		for field in item[field_target]:
			value_target = item[field_target][field]
			value_label = item["label"][field]
			
			
			value_target_binarised = 0
			if value_target >= threshold:
				value_target_binarised = 1
			
			if abs(value_target_binarised - value_label) < 0.1:
				result[field][0] += 1
			else:
				result[field][1] += 1
	
	
	for field in result:
		result_field = result[field]
		result[field] = result_field[0] / (result_field[0] + result_field[1])
				
	return result

def calc_accuracies_multi(dataset, field_targets):
	result = make_empty_probabilities({})
	for field_target in field_targets:
		result_target = calc_accuracies(dataset, field_target)
		# print("DEBUG:result_target")
		# pp.pprint(result_target)
		
		for field, acc in result_target.items():
			result[field][field_target] = acc
	
	return result

def weighted_avg(val_a, val_b, weight_a, weight_b):
	return ((val_a * weight_a) + (val_b * weight_b)) / (weight_a + weight_b)

def argmax_dict(dict_source):
	max_key = None
	max_value = None
	for key, value in dict_source.items():
		if max_value == None or value > max_value:
			max_key = key
			max_value = value
	
	return { "label": max_key, "confidence": max_value }


### INPUT ####################################################################


# If there aren't any labels, then we must be reading them from a file later
if INPUT_LABELS is not None:
	sys.stderr.write(f">>> Read ground-truth labels from {INPUT_LABELS}\n")
	data_labels = hashmapificate(json.load(handle_open(INPUT_LABELS, "r")))
data_nlp = hashmapificate(json.load(handle_open(INPUT_NLP, "r")))
data_vision = hashmapificate(json.load(handle_open(INPUT_VISION, "r")))

# pp.pprint(data_labels)
# pp.pprint(data_nlp)
# pp.pprint(data_vision)


### CONCAT ###################################################################
combined = {}
if INPUT_LABELS is not None:
	for item_id in data_labels:
		combined[item_id] = {
			"id": item_id,
			"label": list2probabilities(data_labels[item_id]["labels"])
		}

for item_id in data_nlp:
	row = combined[item_id] if item_id in combined else None
	if row is None:
		if INPUT_LABELS is None:
			row = { "id": item_id }
			combined[item_id] = row
		else:
			raise Exception(f"Error: INPUT_LABELS doesn't have a row with id {item_id}, but INPUT_NLP does")
	
	
	row["nlp"] = data_nlp[item_id]["predicted_probabilities"]

for item_id in data_vision:
	row = combined[item_id]
	if not row:
		raise Exception(f"Error: INPUT_LABELS doesn't have a row with id {item_id}, but INPUT_VISION does")
	
	row["vision"] = data_vision[item_id]["predicted_probabilities"]


# pp.pprint(combined)

# acc_nlp = calc_accuracies(combined, "nlp")
# acc_vision = calc_accuracies(combined, "vision")

if WEIGHTS_OUT is not None:
	acc_weights = calc_accuracies_multi(combined, ["nlp", "vision"])
	
	handle_weights_out = handle_open(WEIGHTS_OUT, "w")
	handle_weights_out.write(json.dumps(acc_weights))
	handle_weights_out.close()
	
	sys.stderr.write(f">>> Weights written to {WEIGHTS_OUT}\n")
else:
	acc_weights = json.load(handle_open(WEIGHTS_IN, "r"))
	sys.stderr.write(f">>> Weights read from {WEIGHTS_IN}\n")

print("LABEL\tNLP\tVISION")
print("\n".join([
	"\t".join([field] + list(map(lambda x: str(x), dict_acc.values())))
		for field, dict_acc in acc_weights.items()
	]))

# print("DEBUG:accuracies")
# pp.pprint(acc)

predictions = {}

for item_id, item in combined.items():
	acc_combined = make_empty_probabilities()
	for field, value_nlp in item["nlp"].items():
		value_vision = item["vision"][field]
		
		acc_combined[field] = weighted_avg(
			value_nlp, value_vision,
			acc_weights[field]["nlp"],
			acc_weights[field]["vision"]
		)
	
	# prediction = argmax_dict(acc_combined)
	# predictions[item_id] = prediction
	predictions[item_id] = acc_combined
	
		
handle_out = handle_open(OUTPUT, "w")
handle_out.write(json.dumps(predictions))
handle_out.close()

sys.stderr.write(f">>> Results written to {OUTPUT}\n")
