python train.py --dataroot ../data/small_splits1/train --batchSize 8 --imageSize 64 --upSampling 4 --nEpochs 50 --cuda --out ../checkpoints/mse_eph50_run0

echo completed with mse loss. Repeating with lqnorm

python train_lqnorm.py --dataroot ../data/small_splits1/train --batchSize 8 --imageSize 64 --upSampling 4 --nEpochs 50 --cuda --out ../checkpoints/lqnorm_eph50_run0

echo completed with lqnorm loss.