torchrun --standalone --nproc_per_node=8 \
      MARS/train_muon.py \
      config/train_gpt2_large_muon.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12