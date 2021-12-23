python main_cifar.py --batch_size 128 --lr 0.02 --T 0.5 --lambda_u 25 --experiment-name default_0_mcbn_CDGMM --method reg --r 0.5 --num_epochs 300 --resume ./checkpoint/default_0_mcbn_CDGMM/models/15


python main_cifar.py --batch_size 128 --lr 0.02 --T 0.5 --lambda_u 25 --experiment-name default_1_mcbn_CDGMM --method reg --r 0.5 --num_epochs 300 --lambda_c 1 --mcbn True --resume ./checkpoint/default_1_mcbn_CDGMM/models/239


python main_cifar.py --dataset cifar100 --data_path /home/acatalan/Private/datasets/cifar-100-python --batch_size 128 --lr 0.02 --T 0.5 --lambda_u 25 --p_threshold 0.03 --experiment-name baseline_cifar100_2- --method selfsup --r 0.2  --resume /home/acatalan/Private/C2D/checkpoint/baseline_cifar100_20/models/170 --num_epochs 360