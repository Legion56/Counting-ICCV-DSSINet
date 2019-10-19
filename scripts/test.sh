#!/bin/bash
python -u nowtest.py \
		--dataset shanghaiA \
		--model_name CRFVGG \
		--no-preload \
		--no-wait \
		--save \
		--gpus 3\
		--test_batch_size 1 \
		--model_path $1 