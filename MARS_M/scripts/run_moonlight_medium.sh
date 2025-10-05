torchrun --standalone --nproc_per_node=8 \
      train_moonlight.py \
      config/train_gpt2_medium_moonlight.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4