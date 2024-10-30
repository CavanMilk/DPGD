set -euxo pipefail

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main.py --name gpgd_4 --model GPGD --scale 4 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 200 --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main.py --name gpgd_8 --model GPGD --scale 8 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 200 --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main.py --name gpgd_16 --model GPGD --scale 16 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 200 --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2