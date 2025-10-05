# MARS: Unleashing the Power of Variance Reduction for Training Large Models

This repository contains the official code for the paper "MARS-M: When Variance Reduction Meets Matrices".

Authors: [Yifeng Liu](https://scholar.google.com/citations?user=mFvOVkMAAAAJ&hl=zh-CN)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

## 🔔 NEWS

- **[10/05/2025]** Our code is released.

## MARS-M

**MARS-M** is a brand-new optimizer that integrates matrix-level adaptive learning methods (e.g., Muon) into the MARS framework to reduce high stochastic gradient variance in the training process.

The **MARS-M** optimizer is built on **MARS** framework:

$$
\mathbf{c}\_t = \nabla f(\mathbf{x}\_t, \mathbf{\xi}\_t)+\underbrace{{\color{red}\gamma_t} \frac{\beta_{1}}{1-\beta_{1}} \left(\nabla f(\mathbf{x}\_t, \mathbf{\xi}\_t)-\nabla f(\mathbf{x}\_{t-1}, \mathbf{\xi}\_t)\right)}_{\text{scaled gradient correction}}
$$

$$
\tilde{\mathbf{c}}_t = \text{Clip}(\mathbf{c}_t,1) =  \begin{cases}
\frac{\mathbf{c}_t}{\\|\mathbf{c}_t\\|_2} & \text{if } \\|\mathbf{c}_t\\|_2 > 1,\\
\mathbf{c}_t & \text{otherwise}.
\end{cases}
$$

$$
\mathbf{m}\_t = \beta_1 \mathbf{m}\_{t-1} + (1-\beta_{1})\tilde{\mathbf{c}}\_t
$$

$$
\mathbf{x}\_{t+1} = \arg\min_{\mathbf{x} \in \mathbb{R}^d} \left\\{\eta_t \left\langle \mathbf{m}_t, \mathbf{x} \right\rangle + \frac{1}{2} \\|\mathbf{x} - \mathbf{x}\_t
\\|\_{\mathbf{H}_t}^2\right\\}
$$

Here ${\color{red}\gamma_t}$ is a scaling parameter that controls the strength of gradient correction.

Under the **MARS** framework, we propose **MARS-M** that incorporates MARS with matrix-level optimizers (Enable with  `optimizers/mars_m.py`):

$$
\mathbf{O}_t=\text{NewtonSchulz}(\mathbf{m}_t),\qquad 
    \mathbf{x}\_{t+1} =\mathbf{x}\_t-\eta_t(0.2\cdot\mathbf{O}_t\cdot\sqrt{\max(A,B)}+\lambda\mathbf{x}_t).
$$

### **Performance of MARS-M Compared to Baseline of Muon (Moonlight)**

#### Experiments on OpenWebText

In our experiments, gradients are calculated once per sample and per update (**MARS-M**-approx). Performing exact gradient computation with two evaluations per update, as in the exact form of **MARS-M**, can slightly enhance performance but at the cost of doubling the computational expense.

**MARS-M** consistently outperforms [Muon (Moonlight version)](https://arxiv.org/abs/2502.16982) optimizers across GPT-2 models:

| **GPT-2 small**                            | **GPT-2 medium**                            | **GPT-2 large**                            |
| ------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------ |
| <img src="assets/val_small.png" width="350"> | <img src="assets/val_medium.png" width="350"> | <img src="assets/val_large.png" width="350"> |

---

#### Experiments on FineWeb-Edu

Below are the training and validation loss curves for both GPT‑2 Small and GPT‑2 XL when using our MARS-M approach versus [Muon (Moonlight version)](https://arxiv.org/abs/2502.16982) optimizers. As you can see, MARS-M often yields faster convergence and consistently lower losses across different training steps.

| Model                     | **GPT-2 small**                              | **GPT-2 XL**                              |
| ------------------------- | -------------------------------------------------- | ----------------------------------------------- |
| **Train Loss**      | <img src="assets/small_train.png" width="350"> | <img src="assets/xl_train.png" width="350"> |
| **Validation Loss** | <img src="assets/small_val.png" width="350">   | <img src="assets/xl_val.png" width="350">   |

## Training GPT-2 from Scratch:

### Install Dependencies

```
$ pip install torch==2.1.2 transformers==4.33.0 datasets tiktoken numpy==1.26.4 wandb
```

### Data Preparation

Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data following [nanoGPT](https://github.com/karpathy/nanoGPT/):

```
$ python data/openwebtext/prepare.py
```

### **Start Training**

To train a model using the **MARS-M** optimizer, run the following command:

```bash
$ torchrun --standalone --nproc_per_node=8 train_mars_m.py config/${your_config_file}
```

This command initiates the training of a GPT-2 model on the OpenWebText dataset using the **MARS-M** optimizer. All relevant hyperparameters—training, model, and optimizer—are specified in the configuration file (`${your_config_file}`). These parameters can be adjusted directly in the configuration file or through the bash script.

### **Hyperparameter Details**

#### **Model Hyperparameters**:

- **n_layer**: Layers of networks, 12 for GPT2 Small, 24 for GPT2 Medium, 36 for GPT2 Large
- **n_head**: Number of heads, 12 for GPT2 small, 16 for GPT2 Medium, 20 for GPT2 Large
- **n_embd**: Embedding dimension, 768 for GPT2 small, 1024 for GPT2 Medium, 1280 for GPT2 Large

#### **Optimizer Hyperparameters**:

- **`learning_rate`**: Learning rate for the **MARS-M** optimizer.
- **`weight_decay`**: Weight decay for the **MARS-M** optimizer.
- **`beta1`**: momentum for **MARS-M** optimizer.

  - Default: `beta1=0.95, beta2=0.99`
- **`betas_1d`**: Weights for exponential moving average in AdamW optimizer (for 1d parameters).

  - Default: `(0.9, 0.95)`
- **`is_approx`**: Whether to use approximate gradient calculation (**MARS-M**-approx).

  - Default: `True`
- **`gamma`**: The scaling parameter that controls the strength of gradient correction.

  - Default: 0.025

#### **Training Hyperparameters**:

- **`batch_size`**: Mini-batch size per device. (for example GPT-2 Small on an A100 GPU typically uses a batch size of 15.)
- **`gradient_accumulation_steps`**: Gradient accumulation steps to ensure the total effective batch size matches the desired scale. (for example, for a total batch size of 480: $15 \times 4 \times 8 \, \text{GPUs}$.)
- **`schedule`**: learning rate schedule.
  - Default: `cosine`

For more detailed hyperparameter examples, refer to:

- `config/train_gpt2_small_mars_m.py`
- `scripts/run_mars_m_small.sh`

---

### Reproducing Our Results

#### **Reproducing GPT-2 Small (125M) Results**

Training with MARS using

```
$ bash scripts/run_mars_m_small.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars_m.py \
      config/train_gpt2_small_mars_m.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4
```

#### Reproducing GPT2 Medium (355M) Results

Training with MARS using

```
$ bash scripts/run_mars_m_medium.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars.py \
      config/train_gpt2_medium_mars_m.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4
```

#### Reproducing GPT2 Large (770M) Results

Training with MARS using

```
$ bash scripts/run_mars_m_large.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars.py \
      config/train_gpt2_large_mars_m.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
```

#### **Reproducing GPT-2 XL (1.5B) Results on FineWeb-Edu**

```
$ bash scripts/run_mars_m_xl_fw.sh
```

or

```
$ torchrun --standalone --nproc_per_node=8 \
      MARS/train_mars_fw.py \
      config/train_gpt2_xl_mars_m.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12
```

#### Reproducing Baseline Results

To reproduce the Moonlight baseline:

```
bash scripts/run_moonlight_{small/medium/large}.sh
```

Other baselines can be implemented with codes in `../MARS` folder.

Please adjust ``nproc_per_node``, ``batch_size``, and ``gradient_accumulation_steps`` accordingly if you use other hardware setup. Make sure their product equals 480.

#### Hyperparameters for GPT-2 models

|  Model Name  | Model Size | OpenWebText LR | FineWeb-Edu LR | weight decay |
| :----------: | :--------: | :------------: | :------------: | :----------: |
| GPT-2 small |    125M    |      6e-3      |      1e-2      |     1e-1     |
| GPT-2 medium |    355M    |      5e-3      |      5e-3      |     1e-1     |
| GPT-2 large |    770M    |      5e-3      |      5e-3      |     1e-1     |
|   GPT-2 xl   |    1.5B    |       -       |      3e-3      |     1e-1     |

### Customized Training

To build your own training pipeline on other architectures and datasets, use the following template as an example:

```python
import torch
import torch.nn.functional as F
from mars_m import MARS_M

# init model loss function and input data
model = Model()
data_loader = ...

# init the optimizer
muon_params = [p for name, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name]
adamw_params = [p for name, p in model.named_parameters() if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name]
optimizer = MARS_M(muon_params=muon_params, adamw_params=adamw_params, lr=1e-3, betas=(0.9, 0.95), gamma=0.025)

total_bs = len(data_loader)
bs = total_bs * block_size
k = 10
iter_num = -1

# training loop
for epoch in range(epochs):
    for X, Y in data_loader:
        # standard training code
        logits, loss = model(X, Y)
        loss.backward()
        optimizer.step(bs=bs)
        optimizer.zero_grad(set_to_none=True)
        optimizer.update_last_grad()
        iter_num += 1

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AGI-Arena/MARS&type=Date)](https://www.star-history.com/#AGI-Arena/MARS&Date)

## Citation

If you find this repo useful for your research, please consider citing the paper

```tex
TBD
```

## Acknowledgements

This repo is built upon [nanoGPT](https://github.com/karpathy/nanoGPT/), [levanter](https://github.com/stanford-crfm/levanter/) and [Sophia](https://github.com/Liuhong99/Sophia), we thank the authors for their great work!
