python train_outliers.py --dataroot ../data/outliers/train --batchSize 8 --imageSize 64 --upSampling 4 --nEpochs 10 --cuda --out ../checkpoints/mse_eph50_run0_ol --generatorWeights ../checkpoints/mse_eph50_run0/2018-06-23_16:23:58.681659/generator_final.pth --discriminatorWeights ../checkpoints/mse_eph50_run0/2018-06-23_16:23:58.681659/discriminator_final.pth

echo completed with mse loss. Repeating with lqnorm

python train_outliers_lqnorm.py --dataroot ../data/outliers/train --batchSize 8 --imageSize 64 --upSampling 4 --nEpochs 10 --cuda --out ../checkpoints/lqnorm_eph50_run0_ol --generatorWeights ../checkpoints/lqnorm_eph50_run0/2018-06-23_18:10:40.480795/generator_final.pth --discriminatorWeights ../checkpoints/lqnorm_eph50_run0/2018-06-23_18:10:40.480795/discriminator_final.pth

echo completed with lqnorm loss.