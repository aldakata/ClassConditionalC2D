python main_cifar.py --batch_size 128 --lr 0.002 --T 0.7 --lambda_u 25 --experiment-name 10_class_unc --method reg --r 0.5 --mcdo True --dropout True --resume /home/acatalan/Private/C2D/checkpoint/0.1_class_unc/models/9 --num_epochs 100 --lambda_c 10

echo Finished MCDO | 10 lambda_c 100 epochs

python main_cifar.py --batch_size 128 --lr 0.002 --T 0.7 --lambda_u 25 --experiment-name 1_class_unc --method reg --r 0.5 --mcdo True --dropout True --resume /home/acatalan/Private/C2D/checkpoint/0.1_class_unc/models/9 --num_epochs 300 --lambda_c 1

echo Finished MCDO | 1 lambda_c 60 epochs nois 0.9