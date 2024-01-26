#!/usr/bin/env bash

set -x;

root="/tmp/x/SemEval/";
out="/tmp/x/";

labels="${root}/gold labels/dev_subtask2b_en.json";

dev_nlp="${root}/subtask2b_dev_pred_nlp.json";
dev_vision="${root}/subtask2b_dev_pred_clip.json";
test_nlp="${root}/subtask2b_test_pred_nlp.json";
test_vision="${root}/subtask2b_test_pred_clip.json";

dev_weights="${out}/subtask2b_dev_FUSION_WEIGHTS.json";


# Run the dev set through + generate weights
echo "============ DEV ============" >&2;
INPUT_LABELS="${labels}" INPUT_NLP="${dev_nlp}" INPUT_VISION="${dev_vision}" OUTPUT="${out}/subtask2b_dev_pred_FUSION.json" WEIGHTS_OUT="${dev_weights}" ./subtask2b.py | column -t -s "	"

# Use weights to run on test set
echo "============ TEST ============" >&2;
INPUT_NLP="${test_nlp}" INPUT_VISION="${test_vision}" OUTPUT="${out}/subtask2b_test_pred_FUSION.json" WEIGHTS_IN="${dev_weights}" ./subtask2b.py | column -t -s "	"
