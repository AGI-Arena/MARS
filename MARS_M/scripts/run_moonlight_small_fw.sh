torchrun --standalone --nproc_per_node=8 \
      train_moonlight_fw.py \
      config/train_gpt2_small_moonlight.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4
