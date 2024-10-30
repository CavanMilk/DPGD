set -euxo pipefail

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name denoise_gpgd_4 --model GPGD --dataset NoisyMiddlebury --scale 4 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name denoise_gpgd_8 --model GPGD --dataset NoisyMiddlebury --scale 8 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=9 python main.py --test --checkpoint best --name denoise_gpgd_16 --model GPGD --dataset NoisyMiddlebury --scale 16 --interpolation bicubic --data_root ./data/noisy_depth/middlebury --batched_eval --report_per_image