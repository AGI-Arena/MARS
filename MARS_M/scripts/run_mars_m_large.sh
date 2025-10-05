torchrun --standalone --nproc_per_node=8 \
      train_mars_m.py \
      config/train_gpt2_large_mars_m.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12