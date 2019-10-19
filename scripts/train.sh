#!/bin/bash
python -u nowtrain.py \
		--model CRFVGG_prune\
		--dataset shanghaiA \
		--no-save \
		--no-visual \
		--save_interval 2000 \
		--no-preload \
		--batch_size 12 \
		--loss NORMMSSSIM \
		--lr 0.00001 \
		--gpus 3 \
		--epochs 300
