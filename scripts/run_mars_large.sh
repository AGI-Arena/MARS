torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars.py \
      config/train_gpt2_large_mars.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12