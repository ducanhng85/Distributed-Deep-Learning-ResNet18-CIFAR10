# Distributed Deep Learning with ResNet-18 on CIFAR-10

**Comparing single-GPU and multi-GPU training with PyTorch DistributedDataParallel (DDP)**

A course research project by **Duc Anh Nguyen** at the **University of Central Florida**.

This project investigates the performance of fine-tuning a ResNet-18 image classifier on CIFAR-10 using a single GPU versus two GPUs with PyTorch DistributedDataParallel (DDP). The experiments examine training time, classification accuracy, and the practical tradeoffs of distributed training on high-performance computing resources.

## Research question

**Can distributed training reduce training time while maintaining comparable classification accuracy?**

The project evaluates this question using a ResNet-18 model fine-tuned on CIFAR-10 in two computing environments: Google Colab and UCF's ARCC Newton GPU cluster.

## Results

| Metric | Single-GPU baseline | Two-GPU DDP |
| --- | ---: | ---: |
| Training time | 482.60 seconds | 123.23 seconds |
| CIFAR-10 test accuracy | 84.95% | 84.98% |
| Observed time difference | Reference | **3.91× faster** |

The distributed configuration finished the reported training run in less time while producing nearly identical test accuracy.

> **Important limitation:** The two experiments used **different GPU hardware and CPU allocations**. The observed 3.91× speed difference is a comparison of the complete experimental environments, **not a controlled measure of the speedup caused by DDP alone**. A controlled scaling study would compare one and two matched GPUs under the same hardware and data-loading conditions.

## Technical stack

| Category | Tools |
| --- | --- |
| Programming | Python |
| Deep learning | PyTorch, torchvision |
| Model | ResNet-18 (ImageNet-1K pretrained weights) |
| Dataset | CIFAR-10 |
| Distributed computing | PyTorch DDP, `torchrun`, `DistributedSampler`, NCCL |
| Optimization | AdamW, cross-entropy loss, cosine learning-rate scheduler |
| Computing environments | UCF ARCC Newton GPU cluster, Google Colab |

## Methodology

### 1. Dataset preparation

- Used **CIFAR-10**, a 10-class image classification dataset with 50,000 training images and 10,000 test images.
- Converted images into tensors and applied normalization.
- Used PyTorch `DataLoader` for minibatch processing.
- Used `DistributedSampler` to partition training data across GPU processes in the DDP configuration.

### 2. Model architecture and training

- Initialized **ResNet-18** with pretrained ImageNet-1K weights.
- Replaced the final classification layer with a **10-class output layer**.
- Fine-tuned using **AdamW**, cross-entropy loss, and cosine annealing of the learning rate.
- Recorded training time and classification performance.

### 3. Single-GPU baseline

Trained with a standard PyTorch loop using a single GPU, including forward propagation, loss calculation, backpropagation, and parameter updates.

### 4. Two-GPU distributed training

- Launched two training processes with `torchrun --nproc_per_node=2`.
- Assigned each process to an individual GPU.
- Partitioned the dataset using `DistributedSampler`.
- Used PyTorch `DistributedDataParallel` and the **NCCL** communication backend for synchronized gradient updates.
- Performed validation and saved model checkpoints using the rank-0 process.

### 5. Evaluation

Compared **training time**, **test accuracy**, and **accuracy-per-epoch behavior**. The final report also discusses data-loading constraints, GPU synchronization, and how unequal compute resources affect interpretation of the results.

## Key takeaways

- Distributed training can be useful for accelerating deep learning workloads.
- In these experiments, both configurations achieved approximately **85% test accuracy**.
- Performance comparisons must account for more than GPU count: CPU availability, data loading, GPU model, and communication overhead all matter.
- A useful next step is a controlled benchmark on the same GPU hardware, with repeated trials and detailed utilization measurements.

## Project files

This repository will contain the course report and supporting project files. The implementation folders and report will be uploaded separately.

**Report:** *Distributed Deep Learning with ResNet-18 on CIFAR-10 Using PyTorch DDP and GPU Acceleration*.

This work is an **academic course project presented in a research-paper format**; it is **not a peer-reviewed publication**.

---

**Author:** Duc Anh Nguyen  
**Institution:** University of Central Florida  
**Research interests:** Distributed systems, large-scale data processing, high-performance computing, and machine learning
