set -euxo pipefail

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_4 --model GPGD --dataset Middlebury --scale 4 --interpolation bicubic --data_root ./data/depth_enhance/01_Middlebury_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main.py --test --checkpoint ./workspace/checkpoint/gpgd_8.pth --name gpgd_8 --model GPGD --dataset Middlebury --scale 8 --interpolation bicubic --data_root ./data/depth_enhance/01_Middlebury_Dataset
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_16 --model GPGD --dataset Middlebury --scale 16 --interpolation bicubic --data_root ./data/depth_enhance#/01_Middlebury_Dataset

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_4 --model GPGD --dataset Lu --scale 4 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main.py --test --checkpoint ./workspace/checkpoint/gpgd_8.pth --name gpgd_8 --model GPGD --dataset Lu --scale 8 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_16 --model GPGD --dataset Lu --scale 16 --interpolation bicubic --data_root ./data/depth_enhance/03_RGBD_Dataset

#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_4 --model GPGD --dataset NYU --scale 4 --interpolation bicubic --data_root ./data/nyu_labeled#
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main.py --test --checkpoint ./workspace/checkpoint/gpgd_8.pth --name gpgd_8 --model GPGD --dataset NYU --scale 8 --interpolation bicubic --data_root ./data/nyu_labeled
#OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py --test --checkpoint best --name gpgd_16 --model GPGD --dataset NYU --scale 16 --interpolation bicubic --data_root ./data/nyu_labeled
