wandb_log = True
wandb_project = 'mars'
wandb_run_name='gpt2-medium-mars-100k'

batch_size = 15
block_size = 1024
gradient_accumulation_steps = 4

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
scale_attn_by_inverse_layer_idx = True

# this makes total number of tokens be ~50B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = 'mars'
learning_rate = 3e-3 # max learning rate
weight_decay = 1e-2
beta1 = 0.95
beta2 = 0.99
lr_1d=1.5e-3
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
min_lr = 6e-5 

compile = True

out_dir = 'out_medium_mars_100k'
gamma=0.025
