set -euxo pipefail

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=9 python main.py --name denoise_gpgd_4 --model GPGD --scale 4 --noisy --sample_q 30720 --input_size 256 --train_batch 1 --epoch 500 --eval_interval 10 --lr 0.0002  --lr_step 60 --lr_gamma 0.3
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main.py --name denoise_gpgd_8 --model GPGD --scale 8 --noisy --sample_q 30720 --input_size 256 --train_batch 1 --epoch 500 --eval_interval 10 --lr 0.0002  --lr_step 60 --lr_gamma 0.3
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main.py --name denoise_gpgd_16 --model GPGD --scale 16 --noisy --sample_q 30720 --input_size 256 --train_batch 1 --epoch 200 --eval_interval 10 --lr 0.0001  --lr_step 60 --lr_gamma 0.2    