#!/bin/bash
split=`for i in {001..300};do echo -e "${i},\c";done`
python -u nowtest.py \
		--dataset UCF_ECCV_Crop \
		--model_name CRFVGG \
		--no-preload \
		--no-wait \
		--save \
		--gpus 3\
		--test_batch_size 4 \
		--test_fixed_size 512 \
		--model_path $1 \
