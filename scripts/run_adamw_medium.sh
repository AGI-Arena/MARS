torchrun --standalone --nproc_per_node=8 \
      MARS/train_adamw.py \
      config/train_gpt2_medium_adamw.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4