python main_cifar.py --batch_size 128 --lr 0.02 --T 0.5 --lambda_u 25 --experiment-name default_0_mcbn_CDGMM --method reg --r 0.5 --num_epochs 300 --resume ./checkpoint/default_0_mcbn_CDGMM/models/15


python main_cifar.py --batch_size 128 --lr 0.02 --T 0.5 --lambda_u 25 --experiment-name default_1_mcbn_CDGMM --method reg --r 0.5 --num_epochs 300 --lambda_c 1 --mcbn True --resume ./checkpoint/default_1_mcbn_CDGMM/models/239