import torch
from encoders import PointNetCloudEncoder

pc_enc_init_n_channels = 3
pc_enc_init_n_features = 64
pc_enc_n_features = 128

a = torch.rand(100, 3)
#print(a)
output = PointNetCloudEncoder(a, pc_enc_init_n_channels, pc_enc_init_n_features)
print(output)