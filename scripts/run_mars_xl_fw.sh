torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars_fw.py \
      config/train_gpt2_xl_mars.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
