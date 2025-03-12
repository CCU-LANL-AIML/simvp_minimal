# This file was modified by multiple contributors.
# It includes code originally from the OpenSTL project (https://github.com/chengtan9907/OpenSTL).
# The original code is licensed under the Apache License, Version 2.0.

method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 8
hid_T = 16
N_T = 2
N_S = 2
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = 'onecycle'

# EMAHook = dict(
#     momentum=0.999,
#     priority='NORMAL',
# )