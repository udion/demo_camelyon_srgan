python test.py --dataroot ../data/small_splits1/test --batchSize 4 --imageSize 64 --upSampling 4 --cuda --generatorWeights ../checkpoints/mse_eph50_run0/2018-06-23_16:23:58.681659/generator_final.pth --discriminatorWeights ../checkpoints/mse_eph50_run0/2018-06-23_16:23:58.681659/discriminator_final.pth --out mse_eph50_run0

echo outputs of mse finished

python test.py --dataroot ../data/small_splits1/test --batchSize 4 --imageSize 64 --upSampling 4 --cuda --generatorWeights ../checkpoints/lqnorm_eph50_run0/2018-06-23_18:10:40.480795/generator_final.pth --discriminatorWeights ../checkpoints/lqnorm_eph50_run0/2018-06-23_18:10:40.480795/discriminator_final.pth --out lqnorm_eph50_run0

echo outputs of lqnorm finished