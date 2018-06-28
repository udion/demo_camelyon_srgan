python test.py --dataroot ../data/outliers/test --batchSize 4 --imageSize 64 --upSampling 4 --cuda --generatorWeights ../checkpoints/mse_eph50_run0_ol/2018-06-23_20:18:59.992440/generator_final.pth --discriminatorWeights ../checkpoints/mse_eph50_run0_ol/2018-06-23_20:18:59.992440/discriminator_final.pth --out mse_eph50_run0_ol

echo outputs of mse finished

python test.py --dataroot ../data/outliers/test --batchSize 4 --imageSize 64 --upSampling 4 --cuda --generatorWeights ../checkpoints/lqnorm_eph50_run0_ol/2018-06-23_20:44:09.365335/generator_final.pth --discriminatorWeights ../checkpoints/lqnorm_eph50_run0_ol/2018-06-23_20:44:09.365335/discriminator_final.pth --out lqnorm_eph50_run0_ol

echo outputs of lqnorm finished