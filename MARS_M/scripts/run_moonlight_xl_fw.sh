torchrun --standalone --nproc_per_node=8 \
      train_adanw_fw.py \
      config/train_gpt2_large_moonlight.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
