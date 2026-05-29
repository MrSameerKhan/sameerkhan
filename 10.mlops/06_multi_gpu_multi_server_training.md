# 06 — Multi-GPU and Multi-Server ML Training — End to End

> Training large models that don't fit on a single GPU, or training faster by using many GPUs in parallel.

Running example: Fine-tuning LLaMA-2-7B on a sentiment classification dataset. Model size: 7B params × 4 bytes (fp32) = 28 GB — won't fit on a single 24GB GPU. Training data: 500K examples. Goal: train in hours, not weeks.

---

## 0. Why Multi-GPU Training?

```
Single GPU (A100 80GB):
  Model: 7B params × 2 bytes (bf16) = 14 GB weights
  Optimizer states: 14 GB × 2 (Adam momentum + variance) = 28 GB
  Gradients: 14 GB
  Activations (batched): ~8 GB
  Total needed: ~64 GB → doesn't fit in 80GB (barely!)

  Training time: 500K examples × 3 epochs / 8 per batch = 187K steps
                 × 2s/step = 4.3 days

4 × A100 80GB (one machine):
  Split model/data across 4 GPUs
  Training time: ~1.1 days

16 × A100 80GB (4 machines × 4 GPUs):
  Training time: ~7 hours

Why it matters: iteration speed. Getting feedback in 7h vs 4.3 days = 15× faster research.
```

---

## 1. Parallelism Strategies

### 1.1 Data Parallel (DP) — Old Way, Don't Use

```
Each GPU gets a copy of the FULL model.
Each GPU processes a different batch of data.
Gradients averaged across all GPUs.

Problem: uses Python GIL (one Python process), doesn't scale beyond 1 machine.
Replace with: DDP.
```

### 1.2 Distributed Data Parallel (DDP) — Standard

```
Each GPU gets a copy of the FULL model.
Each GPU processes a different batch (different slice of data).
After backward pass: gradients averaged across all GPUs via all_reduce.
All GPUs end up with identical updated weights.

Constraint: model must fit on a SINGLE GPU.
Best for: smaller models (BERT, GPT-2) that fit per GPU.

Example: 4 GPUs, batch size = 32 total
  GPU 0: batch[0:8]
  GPU 1: batch[8:16]
  GPU 2: batch[16:24]
  GPU 3: batch[24:32]
  All GPUs compute gradients + all_reduce averages them + all GPUs update weights

Throughput: 4× single GPU (ideally)
```

### 1.3 Model Parallel (Tensor Parallel) — Split One Layer Across GPUs

```
Problem: model too large for a single GPU.
Solution: split individual layers (weight matrices) across GPUs.

Example: Linear layer W is [4096 × 4096]
  GPU 0: W[:, :2048]   ← left half of weight matrix
  GPU 1: W[:, 2048:]   ← right half of weight matrix
  Both compute partial output + communicate to combine results

Used in: Megatron-LM, tensor_parallel in HuggingFace.
Communication overhead: high (every forward pass needs all_reduce for each layer).
Best for: very large models on fast inter-GPU interconnects (NVLink).
```

### 1.4 Pipeline Parallel — Split Layers Across GPUs

```
Different LAYERS on different GPUs.
Mini-batches "flow" through the pipeline.

GPU 0: Layers 1-8    (input processing)
GPU 1: Layers 9-16   (middle)
GPU 2: Layers 17-24  (middle)
GPU 3: Layers 25-32  (output, loss)

Micro-batching: split batch into micro-batches. While GPU 1 processes micro-batch 2,
GPU 0 already started micro-batch 3. Pipeline fills up → GPUs stay busy.

Drawback: "pipeline bubble" at start and end (some GPUs idle).
Best for: very deep models, combined with DDP across nodes.
```

### 1.5 FSDP — Fully Sharded Data Parallel (PyTorch Native)

```
FSDP = DDP + model sharding.

Each GPU stores only 1/N of the model parameters.
Before a forward pass: gather parameters from all GPUs.
After backward pass: shard gradients, update sharded parameters.

Saves GPU memory: each GPU only holds 1/N of params at rest.
Tradeoff: more inter-GPU communication during forward/backward.

Best for: models too large for DDP but want to stay in PyTorch.
LLaMA-2-7B with 4 GPUs: each GPU holds 1.75B params = 3.5 GB (bf16).
```

### 1.6 DeepSpeed ZeRO — Most Practical for LLMs

```
ZeRO = Zero Redundancy Optimizer. Shards across 3 levels:

ZeRO Stage 1: Shard optimizer states only
  Each GPU: full model params + full gradients + 1/N optimizer states
  Memory saved: 4× (optimizer states are the biggest)

ZeRO Stage 2: Shard optimizer states + gradients
  Each GPU: full model params + 1/N gradients + 1/N optimizer states
  Memory saved: 8×

ZeRO Stage 3: Shard everything (params + gradients + optimizer states)
  Each GPU: 1/N params + 1/N gradients + 1/N optimizer states
  Memory saved: N× (number of GPUs)
  Tradeoff: more communication

ZeRO-Offload: move optimizer states to CPU RAM (saves GPU, slower)
ZeRO-Infinity: move everything to CPU/NVMe (slowest, enables trillion-param training)

Most used: ZeRO Stage 2 (good balance) or Stage 3 for very large models.
```

Memory comparison for 7B training, 8 GPUs:

```
| Strategy            | GPU mem per card | Communication | Speed   |
|---------------------|-----------------|---------------|---------|
| DDP (model copy)    | 80 GB (OOM!)    | Low           | Fastest |
| FSDP                | 12 GB           | Medium        | Fast    |
| ZeRO Stage 2        | 15 GB           | Medium        | Fast    |
| ZeRO Stage 3        | 8 GB            | High          | Fast    |
| ZeRO Stage 3+Offload| 6 GB            | High + PCIe   | Slow    |
```

---

## 2. Hardware Prerequisites

### 2.1 Single Machine Multi-GPU

```
Minimum for 7B model training:
  GPUs: 4 × NVIDIA A100 40GB or 2 × A100 80GB
  CPU:  32 cores (data loading runs on CPU)
  RAM:  256 GB (ZeRO-Offload moves states to CPU RAM)
  Disk: 2 TB NVMe SSD (checkpoints + dataset)
  GPU interconnect: NVLink 3.0 (A100) = 600 GB/s bidirectional

NVLink vs PCIe:
  PCIe 4.0:  64 GB/s bidirectional
  NVLink:   600 GB/s bidirectional (9× faster!)

Why it matters: ZeRO Stage 3 does heavy inter-GPU communication.
  With NVLink: training is near-linear scaling efficiency
  With PCIe only: GPU waiting for data → poor scaling efficiency

Check NVLink:
  nvidia-smi topo -m   # shows GPU-to-GPU interconnect type
  # NV4 = 4 NVLink links, PIX = PCIe only
```

### 2.2 Multi-Machine Multi-GPU

```
Example: 4 servers × 4 A100 80GB = 16 GPUs total

Per server:
  GPUs: 4 × A100 80GB
  CPU:  64 cores
  RAM:  512 GB
  Disk: 6 TB NVMe SSD (shared dataset via NFS, or local copy)
  NIC:  100 Gbps InfiniBand (critical for inter-server communication)

Inter-server networking:
  InfiniBand (IB):  200 Gbps, ~1µs latency — standard for HPC training
  Ethernet 100 GbE: 200 Gbps, ~10µs latency — 10× slower, but cheaper
  RoCE (RDMA over Ethernet): IB performance over Ethernet — compromise

Rule: use InfiniBand if budget allows. For 4+ nodes training,
Ethernet can become the bottleneck with ZeRO Stage 3.

Check inter-server bandwidth:
  # On server 1:
  ib_send_bw -d mlx5_0 -i 1              # server-side
  # On server 2:
  ib_send_bw -d mlx5_0 -i 1 192.168.1.100   # client-side
  # Result: 95,000 MB/s = 95 GB/s on IB
```

---

## 3. Network Setup for Multi-Server Training

```bash
# — On ALL servers ——————————————————————————

# 1. Set hostnames
sudo hostnamectl set-hostname node-0  # on server 1
sudo hostnamectl set-hostname node-1  # on server 2
sudo hostnamectl set-hostname node-2  # on server 3
sudo hostnamectl set-hostname node-3  # on server 4

# 2. /etc/hosts — all nodes know each other
sudo tee -a /etc/hosts << 'EOF'
192.168.1.10  node-0
192.168.1.11  node-1
192.168.1.12  node-2
192.168.1.13  node-3
EOF

# 3. Passwordless SSH between nodes (required for torchrun/deepspeed launcher)
# On node-0: generate key pair
ssh-keygen -t ed25519 -f ~/.ssh/cluster_key -N ""
# Copy public key to ALL other nodes
for node in node-1 node-2 node-3; do
  ssh-copy-id -i ~/.ssh/cluster_key.pub ubuntu@$node
done
# Test: node-0 must SSH to all others without password
ssh -i ~/.ssh/cluster_key ubuntu@node-1 "hostname"
ssh -i ~/.ssh/cluster_key ubuntu@node-2 "hostname"
ssh -i ~/.ssh/cluster_key ubuntu@node-3 "hostname"

# 4. NFS shared storage (dataset + checkpoints accessible from all nodes)
# On node-0 (NFS server):
sudo apt install -y nfs-kernel-server
sudo mkdir -p /data/shared
sudo tee -a /etc/exports << 'EOF'
/data/shared 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)
EOF
sudo exportfs -a
sudo systemctl enable nfs-kernel-server --now

# On all other nodes (NFS clients):
sudo apt install -y nfs-common
sudo mkdir -p /data/shared
echo "node-0:/data/shared /data/shared nfs defaults,nofail 0 0" | sudo tee -a /etc/fstab
sudo mount -a
df -h | grep shared

# 5. NCCL environment variables (tune for your network)
export NCCL_DEBUG=INFO               # verbose logs (use during debugging)
export NCCL_IB_DISABLE=0             # enable InfiniBand
export NCCL_IB_HCA=mlx5_0           # specify IB device (check with: ibstat)
export NCCL_SOCKET_IFNAME=eth0       # which NIC to use for fallback
export NCCL_TREE_THRESHOLD=0         # use ring allreduce (better for large tensors)

# Add to ~/.bashrc on all nodes
```

---

## 4. Single Machine Multi-GPU: DDP with torchrun

### 4.1 Training Script

```python
# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from dataset import load_dataset
import logging

# — Distributed setup ————————————————————
def setup():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")  # NCCL: best for GPU-GPU communication
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0   # rank 0 = main process (does logging, saving)

# — Main training function ————————————————
def train():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])   # GPU index on this machine
    global_rank = dist.get_rank()                # GPU index across all machines
    world_size = dist.get_world_size()           # total number of GPUs

    if is_main_process():
        logging.basicConfig(level=logging.INFO)
        print(f"Training with {world_size} GPUs")

    # — Model ————————————————————————————
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(local_rank)   # move to this GPU

    # Wrap in DDP: handles gradient synchronization automatically
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # — Dataset with DistributedSampler ——
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw_dataset = load_dataset("imdb")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
    dataset = raw_dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,             # per-GPU batch size
        sampler=sampler,          # NOT shuffle=True when using DistributedSampler
        num_workers=4,            # CPU workers for data loading
        pin_memory=True,          # faster CPU → GPU transfer
    )

    # — Optimizer and Scheduler ——————————
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    num_training_steps = len(dataloader) * 3  # 3 epochs
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=100,
                               num_training_steps=num_training_steps)

    # — Training Loop ————————————————————
    for epoch in range(3):
        sampler.set_epoch(epoch)   # ensures different shuffling each epoch
        model.train()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()   # DDP automatically averages gradients across GPUs here

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Log only from main process (avoid duplicate logs)
        if is_main_process() and step % 100 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    # Save checkpoint from main process only
    if is_main_process():
        model.module.save_pretrained(f"/data/shared/checkpoints/epoch-{epoch}")
        tokenizer.save_pretrained(f"/data/shared/checkpoints/epoch-{epoch}")
        print(f"Checkpoint saved: epoch-{epoch}")

    cleanup()

if __name__ == "__main__":
    train()
```

### 4.2 Launch Command

```bash
# Single machine, 4 GPUs
torchrun \
  --standalone \        # single node
  --nproc_per_node=4 \  # 4 GPUs on this machine
  train_ddp.py

# torchrun sets environment variables automatically:
#   LOCAL_RANK  = GPU index on this machine (0, 1, 2, 3)
#   RANK        = global GPU index (same as LOCAL_RANK for single node)
#   WORLD_SIZE  = total GPUs (4)
#   MASTER_ADDR = localhost
#   MASTER_PORT = random available port

# Monitor all GPUs:
watch -n 1 nvidia-smi
```

---

## 5. Multi-Machine Multi-GPU: torchrun Multi-Node

```bash
# — On NODE-0 (master) ——————————————————
torchrun \
  --nnodes=4 \              # 4 machines total
  --nproc_per_node=4 \      # 4 GPUs per machine
  --node_rank=0 \           # this is node 0
  --master_addr=192.168.1.10 \  # node-0's IP (all nodes must reach this)
  --master_port=29500 \     # any free port (same on all nodes)
  train_ddp.py

# — On NODE-1 ————————————————————————————
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=1 \
  --master_addr=192.168.1.10 --master_port=29500 train_ddp.py

# — On NODE-2 ————————————————————————————
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=2 --master_addr=192.168.1.10 \
  --master_port=29500 train_ddp.py

# — On NODE-3 ————————————————————————————
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=3 --master_addr=192.168.1.10 \
  master_port=29500 train_ddp.py

# Total: 16 GPUs, world_size=16
# Effective batch size: 8 per GPU × 16 GPUs = 128 total
```

What happens internally:

```
Initialization:
  All 16 processes contact master (node-0:29500)
  Each gets its RANK (0-15) and WORLD_SIZE (16)
  NCCL builds a communication ring across all 16 GPUs

Forward pass (step 1):
  GPUs 0-3  (node-0): batch[0:32]
  GPUs 4-7  (node-1): batch[32:64]
  GPUs 8-11 (node-2): batch[64:96]
  GPUs 12-15(node-3): batch[96:128]
  Each GPU computes loss on its micro-batch independently

Backward pass:
  Each GPU computes gradients for its micro-batch

All-reduce (critical step):
  NCCL performs ring all-reduce across all 16 GPUs
  Ring topology: 0→1→2→...→15→0→1→...→15→0
  Each GPU sends its gradient, accumulates received gradients
  After one full ring pass: every GPU has the average gradient
  Bandwidth used: 2 × (N-1)/N × grad_size per GPU (ring efficiency)

Optimizer step:
  All 16 GPUs update weights identically (same averaged gradient, same LR)
  All GPUs stay in sync automatically
```

---

## 6. FSDP (Fully Sharded Data Parallel)

Use when model doesn't fit per GPU even for DDP.

```python
# train_fsdp.py
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, \
    MixedPrecision, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaDecoderLayer
import functools

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train_fsdp():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])

    # — Model: load on CPU first (can't load 7B model directly on GPU) ——
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,   # bf16 to save memory
    )

    # FSDP wrapping policy: shard at decoder layer boundaries
    # Each LlamaDecoderLayer is sharded independently
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Mixed precision: keep weights in bf16, compute in bf16, reduce in fp32
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,    # use fp32 for gradient reduction
        buffer_dtype=torch.bfloat16,
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO Stage 3 equivalent
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # prefetch next layer's params
    )
    # After FSDP wrapping: each GPU holds 1/N of the model parameters
    # For 7B model on 4 GPUs: each GPU holds ~1.75B params = ~3.5 GB in bf16

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save: FSDP requires gathering shards from all GPUs before saving
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if dist.get_rank() == 0:
        torch.save(cpu_state, "/data/shared/checkpoints/fsdp_final.pt")

# Launch FSDP (same as DDP)
# torchrun --standalone --nproc_per_node=4 train_fsdp.py
```

---

## 7. DeepSpeed ZeRO — Most Practical for LLMs

### 7.1 DeepSpeed Config

```json
// ds_config_zero2.json — ZeRO Stage 2
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,

  "fp16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "weight_decay": 0.01,
      "betas": [0.9, 0.999]
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-5,
      "warmup_num_steps": 100,
      "total_num_steps": 10000
    }
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

```json
// ds_config_zero3.json — ZeRO Stage 3 (for very large models)
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,

  "fp16": {"enabled": true},

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true,

    "offload_optimizer": {          // ZeRO-Offload: move optimizer to CPU
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {              // ZeRO-Infinity: move params to CPU/NVMe
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### 7.2 Training Script with DeepSpeed

```python
# train_deepspeed.py
import deepspeed
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def train():
    # deepspeed.init_distributed() handles dist.init_process_group()
    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])

    # — Model: load on CPU ——————————————————
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
    )

    # — DeepSpeed engine: wraps model, optimizer, scheduler ——
    model_engine, optimizer, dataloader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config="ds_config_zero2.json",
    )

    # model_engine.device = the local GPU

    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            # Move batch to current device
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)

            # Forward
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward (DeepSpeed handles gradient accumulation, scaling, clipping)
            model_engine.backward(loss)

            # Optimizer step (DeepSpeed handles ZeRO communication)
            model_engine.step()

            if model_engine.global_rank == 0 and step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    # Save checkpoint
    model_engine.save_checkpoint("/data/shared/checkpoints/", tag=f"epoch-{epoch}")

if __name__ == "__main__":
    train()
```

### 7.3 Launch Commands

```bash
# — Single machine, 4 GPUs, with DeepSpeed ——
deepspeed --num_gpus=4 train_deepspeed.py

# — Multi-machine, 16 GPUs ——————————————————
# Create hostfile
cat > hostfile << 'EOF'
192.168.1.10 slots=4
192.168.1.11 slots=4
192.168.1.12 slots=4
192.168.1.13 slots=4
EOF

deepspeed \
  --hostfile hostfile \
  --master_addr 192.168.1.10 \
  --master_port 29500 \
  train_deepspeed.py

# — With HuggingFace Trainer (easiest integration) ——
# Just add deepspeed config to TrainingArguments — no code changes needed
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="/data/shared/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    bf16=True,
    deepspeed="ds_config_zero2.json",   # ← just add this line
    logging_steps=100,
    save_strategy="epoch",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## 8. HuggingFace Accelerate — Simplest Multi-GPU

Accelerate abstracts DDP/FSDP/DeepSpeed behind one interface.

```bash
# Install
pip install accelerate

# Configure (interactive setup)
accelerate config
# Prompts:
#   - Multi-GPU? yes
#   - Number of machines? 1 (or 4)
#   - GPUs per machine? 4
#   - Mixed precision? bf16
#   - DeepSpeed? yes/no
#   - FSDP? yes/no
# Creates ~/.cache/huggingface/accelerate/default_config.yaml
```

```python
# train_accelerate.py — minimal changes to existing single-GPU code
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# accelerator.prepare() wraps everything for multi-GPU
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for epoch in range(3):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)   # instead of loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save from main process only
    if accelerator.is_main_process:
        accelerator.save_model(model, f"/data/shared/checkpoints/epoch-{epoch}")
```

```bash
# Launch (Accelerate reads config automatically)
accelerate launch \
  --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  train_accelerate.py

# Or override on command line
accelerate launch --num_processes=4 --mixed_precision=bf16 train_accelerate.py

# Multi-node
accelerate launch \
  --num_machines=4 \
  --num_processes=16 \
  --main_process_ip=192.168.1.10 \
  --main_process_port=29500 \
  --machine_rank=0 \   # change for each node
  train_accelerate.py
```

---

## 9. Gradient Accumulation — Simulate Large Batches

When GPU memory limits batch size, accumulate gradients over multiple steps.

```python
# Effective batch size = batch_per_gpu × num_gpus × accumulation_steps
# Example: batch=2, 4 GPUs, accumulation=8 → effective batch = 64
ACCUMULATION_STEPS = 8

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / ACCUMULATION_STEPS   # normalize loss

    loss.backward()   # accumulate gradients (don't zero yet)

    if (step + 1) % ACCUMULATION_STEPS == 0:
        # Gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()   # now zero the accumulated gradients

# DDP note: gradients are synchronized EVERY backward() call by default.
# For efficiency, use no_sync() context to skip sync during accumulation steps:

for step, batch in enumerate(dataloader):
    is_last_accumulation_step = (step + 1) % ACCUMULATION_STEPS == 0

    # Only sync gradients on the last accumulation step
    if not is_last_accumulation_step:
        with model.no_sync():   # skip gradient sync
            loss = model(**batch).loss / ACCUMULATION_STEPS
            loss.backward()
    else:
        loss = model(**batch).loss / ACCUMULATION_STEPS
        loss.backward()   # this syncs gradients across all GPUs
        optimizer.step()
        optimizer.zero_grad()
```

---

## 10. Checkpointing Strategy

```python
# Checkpoint every N steps — not just every epoch (training can crash)
SAVE_EVERY_STEPS = 500

for step, batch in enumerate(dataloader):
    # ... training step ...

    if is_main_process() and (step + 1) % SAVE_EVERY_STEPS == 0:
        checkpoint_dir = f"/data/shared/checkpoints/step-{step+1}"

        # For DDP: save model.module (the underlying model, not the DDP wrapper)
        model.module.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Also save optimizer + scheduler state (for exact resume)
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
            },
            f"{checkpoint_dir}/training_state.pt",
        )
        print(f"[Rank 0] Checkpoint saved at step {step+1}")

# Resume from checkpoint
def load_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    model.load_state_dict(torch.load(f"{checkpoint_dir}/pytorch_model.bin"))

    state = torch.load(f"{checkpoint_dir}/training_state.pt")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["step"], state["epoch"]
```

---

## 11. Monitoring Distributed Training

```python
# Per-GPU monitoring during training

# In training loop — log from all GPUs
print(f"[GPU {local_rank}] Step {step} Loss {loss.item():.4f}")
print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB / "
      f"{torch.cuda.get_device_properties(local_rank).total_memory/1e9:.0f}GB")

# W&B multi-GPU logging
import wandb
if is_main_process():
    wandb.init(project="llama-7b-ddp-4gpu")
    wandb.log({"loss": loss.item(), "step": step, "lr": scheduler.get_last_lr()[0]})
```

```bash
# Terminal monitoring — all GPUs at once
watch -n 1 nvidia-smi

# Per-process GPU stats
nvidia-smi pmon -s u   # per-process: PID, GPU util%, memory

# NCCL communication stats (bandwidth between GPUs)
NCCL_DEBUG=INFO torchrun ... train_ddp.py 2>&1 | grep "NCCL"
# Look for: "NCCL INFO Ring... busB0 89.2 GB/s"
# Expected with NVLink: 80-100 GB/s per GPU pair

# Training throughput
# tokens/second = batch_size × seq_len × world_size / step_time
# For LLaMA-7B on 4×A100: target ~15K tokens/second
```

---

## 12. Common Issues and Fixes

```
Problem: NCCL timeout during training
  Error: "NCCL timeout in ncclAllReduce"
  Reason: one GPU is slower than others (data loading bottleneck, different batch sizes)
  Fix:
    1. Check data loading: num_workers≥4 minimum, pin_memory=True
    2. Ensure all ranks have same batch size (DistributedSampler handles this)
    3. Increase NCCL timeout: export NCCL_TIMEOUT=1800
    4. If one node is slow: check network (ib_send_bw)

Problem: CUDA OOM during training
  Error: "RuntimeError: CUDA out of memory"
  Fix sequence:
    1. Reduce per_device_train_batch_size (try halving it)
    2. Increase gradient_accumulation_steps to compensate
    3. Enable gradient checkpointing: model.gradient_checkpointing_enable()
       (recomputes activations during backward instead of storing them — saves 60% memory)
    4. Switch from DDP to FSDP or ZeRO Stage 2/3
    5. Use bf16 instead of fp32

Problem: Training hangs — all GPUs stuck
  Reason: one process is stuck (data loading deadlock, one GPU crashed)
  Fix:
    1. Check all GPU processes: ps aux | grep python
    2. Check NCCL logs: NCCL_DEBUG=INFO to find where it's stuck
    3. If one node is unreachable: ping node-1
    4. Kill all processes: pkill -9 -f train_ddp.py (on all nodes)
    5. Common cause: dataloader with num_workers > 0 + CUDA fork issue
       Fix: add multiprocessing_context="forkserver" to DataLoader

Problem: Loss is NaN after a few steps
  Reason: gradient explosion
  Fix:
    1. Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    2. Reduce learning rate (LLM training: lr=1e-5 to 1e-4 typically)
    3. Switch from fp16 to bf16 (bf16 has wider dynamic range, less overflow)
    4. Check for bad batches (NaN in input data)

Problem: One GPU has much higher memory than others
  Reason: all output/loss computed on one GPU (DataParallel issue)
  Fix: ensure model = DDP(model, device_ids=[local_rank]) ← correct
       model = torch.nn.DataParallel(model)               ← wrong (uses GPU 0 as master)

Problem: Training not scaling linearly (e.g., 2× speed, not 4×)
  Reason: communication bottleneck or data loading bottleneck
  Fix:
    1. Profile: time the forward pass vs all_reduce separately with profiler
    2. If communication: use NVLink (not PCIe), increase batch size (amortize communication)
    3. If data loading: increase num_workers, use memory-mapped datasets
    4. Enable overlapping: DDP find_unused_parameters=False, overlap_comm=True in ZeRO

Problem: Checkpoint loading fails on different world size
  Reason: saved with 4 GPUs, loading with 8 GPUs (full model state, not sharded state)
  Fix: always save from rank 0 only (full model state, not sharded state)
       For FSDP: use FullStateDictConfig with rank0_only=True
       For ZeRO Stage 3: deepspeed.utils.zero_to_fp32.py to convert
```

---

## 13. Scaling Efficiency Table

Ideal: N GPUs = N× single GPU throughput (linear scaling). Reality: communication overhead reduces efficiency.

```
| Setup                   | Throughput | Scaling efficiency | Bottleneck           |
|-------------------------|-----------|-------------------|----------------------|
| 1× A100 (baseline)      | 1.0×      | 100%              | -                    |
| 2× A100, NVLink         | 1.92×     | 96%               | communication ~4%    |
| 4× A100, NVLink         | 3.72×     | 93%               | communication ~7%    |
| 8× A100, NVLink         | 7.2×      | 90%               | communication ~10%   |
| 4 nodes × 4 GPU, IB     | 13.6×     | 85%               | inter-node comm ~15% |
| 4 nodes × 4 GPU, 100GbE | 10.4×     | 65%               | inter-node comm ~35% |
```

Key: InfiniBand is critical for multi-node efficiency. Ethernet training is feasible but noticeably less efficient.

---

## Key Takeaway

```
Multi-GPU training strategies:
  DDP — model fits per GPU, fastest, best for BERT/GPT-2 scale
  FSDP — shards params across GPUs, PyTorch native, good for LLaMA-scale
  DeepSpeed ZeRO — most memory-efficient, best for LLMs (Stage 2 default, Stage 3 for 70B+)

Hardware:
  NVLink for intra-node (9× faster than PCIe)
  InfiniBand for inter-node (10× lower latency than Ethernet)

Setup:
  passwordless SSH, /etc/hosts on all nodes, NFS shared storage, NCCL env vars

Launch:
  torchrun (DDP/FSDP) or deepspeed (ZeRO)

Always:
  DistributedSampler so each GPU sees different data
  Save checkpoints from rank 0 only
  Gradient clipping to prevent NaN loss
  Gradient accumulation to simulate large batch

Monitor:
  nvidia-smi pmon for per-GPU util
  NCCL_DEBUG=INFO for communication issues
  W&B for loss curves across all runs
```
