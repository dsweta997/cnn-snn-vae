"""
CNN-SNN-VAE: Learned Downsampling of Event Data using a Convolutional-Spiking VAE.

Package layout
--------------
layers      -- spiking primitives: SpikeAct, LIFSpike, SampledSpikeAct, td* layers
losses      -- MMD_loss
model       -- VAE (encoder + MMD latent + decoder)
classifier  -- VAEClassifier + per-epoch train/eval helpers
datasets    -- transforms, EventDataset, NMNIST / PokerDVS loader factories
train       -- VAE train_one_epoch / test_one_epoch / train_model
utils       -- AverageMeter, stats helpers, visualisation utilities
"""
