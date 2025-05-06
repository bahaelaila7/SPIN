#!/bin/sh
export PYTHONPATH="$PYTHONPATH:alignment-handbook/src/:trl/"
python spin/reformat.py --data HuggingFaceH4/ultrafeedback_binarized --output_dir data_input/SPIN_iter0/ultrafeedback
python spin/reformat.py --data HuggingFaceH4/cai-conversation-harmless --output_dir data_input/SPIN_iter0/conversation
