#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=8 \
	sample_video.py --video-size 1280 720 --video-length 129 \
     	--infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
    	--flow-reverse --ulysses-degree=8 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node 4 \
	        sample_video.py --video-size 1280 720 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=4 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=2 \
	        sample_video.py --video-size 1280 720 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=2 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=1 \
	        sample_video.py --video-size 1280 720 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=1 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=6 \
	        sample_video.py --video-size 960 960 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=6 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=3 \
	        sample_video.py --video-size 960 960 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=3 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=2 \
	        sample_video.py --video-size 960 960 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=2 --ring-degree=1 --seed 42 --save-path ./results

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=1 \
	        sample_video.py --video-size 1280 720 --video-length 129 \
		        --infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
			        --flow-reverse --ulysses-degree=1 --ring-degree=1 --seed 42 --save-path ./results
