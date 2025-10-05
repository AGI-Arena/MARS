torchrun --standalone --nproc_per_node=8 \
      train_mars_m.py \
      config/train_gpt2_small_mars_m.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4