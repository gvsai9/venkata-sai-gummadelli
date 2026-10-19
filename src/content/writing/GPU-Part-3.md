---
title: "GPUs Part 3"
description: "Some idea on GPU."
pubDate: 2026-10-20
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/gpu.png
thumbnailAlt: "Description of the image"
---
# Part 3 — What Actually Happens Inside the GPU

## From Kernel Launch to GPU Execution

When the CPU launches a GPU kernel, it defines the number of blocks and threads required to execute the operation.

The GPU then:

1. Creates the requested grid of blocks.
2. Assigns blocks to available Streaming Multiprocessors.
3. Divides the threads in each block into warps.
4. Schedules warps for execution.
5. Executes the kernel instructions across the threads.
6. Stores the results in GPU memory.

The CPU launches the work, but the GPU controls how blocks and warps are scheduled internally.

## How Blocks Are Assigned to SMs

A GPU contains multiple Streaming Multiprocessors, or SMs. When a kernel is launched, its blocks are distributed among these SMs.

Each block is assigned to one SM and remains there until all its threads finish execution. A block is not split between multiple SMs.

For example, if a GPU has four SMs and a kernel contains eight blocks, the GPU may initially assign one block to each SM:

```text
SM 0 → Block 0
SM 1 → Block 1
SM 2 → Block 2
SM 3 → Block 3
```

After a block finishes, another waiting block can be assigned to that SM:

```text
SM 0 → Block 4
SM 1 → Block 5
SM 2 → Block 6
SM 3 → Block 7
```

The exact scheduling order is managed by the GPU hardware. Programmers generally do not control which specific SM executes a block.

## What Is an SM (Streaming Multiprocessor)?

A Streaming Multiprocessor is a major processing unit inside a GPU.

An SM contains the hardware required to execute many GPU threads, including:

- Arithmetic processing units.
- Warp schedulers.
- Registers.
- Shared memory.
- Load and store units.
- Special-function units.

A GPU is built by combining many SMs. Each SM executes the blocks assigned to it.

Different GPU architectures may use different names or internal designs, but the basic purpose remains similar: an SM manages and executes groups of parallel threads.

## What Does an SM Actually Do?

An SM receives one or more thread blocks and manages their execution.

Its main responsibilities include:

- Storing the state of active threads.
- Dividing blocks into warps.
- Scheduling ready warps.
- Fetching and issuing instructions.
- Performing arithmetic operations.
- Loading data from memory.
- Storing results.
- Synchronizing threads within a block.

An SM can keep many warps active at the same time. When one warp is waiting for data from memory, the SM can switch to another ready warp.

This ability helps the GPU hide memory-access delays and keep its processing units busy.

## How Threads Are Grouped into Warps

After a block is assigned to an SM, its threads are divided into groups called warps.

On NVIDIA GPUs, a warp contains 32 threads. If a block contains 256 threads, it is divided into:

```text
256 threads ÷ 32 threads per warp = 8 warps
```

The grouping is normally based on consecutive thread IDs:

```text
Warp 0 → Threads 0–31
Warp 1 → Threads 32–63
Warp 2 → Threads 64–95
```

The programmer defines the number of threads in a block, while the hardware organizes those threads into warps.

## Warp-Level Execution

A warp is the basic scheduling unit of many GPU architectures.

The SM selects a ready warp and issues an instruction for it. The threads in that warp execute the instruction on their own data.

For example, if every thread must add two values:

```text
Thread 0 → a + b
Thread 1 → a + b
Thread 2 → a + b
```

The warp executes the same addition instruction, but each thread uses different input values.

The SM switches between available warps to improve utilization. It may select another warp when the current warp is waiting for memory or another resource.

## SIMT: Single Instruction, Multiple Threads

GPU execution is commonly described as SIMT, or **Single Instruction, Multiple Threads**.

Under SIMT:

- Threads execute the same kernel.
- Threads have separate registers and data.
- Threads can follow different control paths.
- The hardware groups threads into warps for execution.

SIMT combines aspects of two models:

- SIMD applies one instruction to multiple data elements.
- Multithreading allows each thread to maintain its own execution state.

This model makes it possible to write programs using individual threads while allowing the hardware to execute them efficiently in groups.

## Why 32 Threads Form a Warp

The 32-thread warp size used by NVIDIA GPUs is a hardware design choice.

A warp of 32 threads provides a balance between:

- Parallel execution.
- Hardware scheduling complexity.
- Register management.
- Instruction issue efficiency.
- Control-flow management.

The warp size is not a universal rule for every GPU architecture. Different GPU vendors or architectures may use different execution-group sizes.

For NVIDIA CUDA programming, however, 32 threads per warp is an important assumption. Block sizes that are multiples of 32 often use hardware resources more efficiently because they avoid partially filled warps.

For example:

```text
Block size: 256 threads
Warps: 8 complete warps

Block size: 250 threads
Warps: 7 complete warps + 1 partially filled warp
```

Inactive threads in a partially filled warp still consume some execution resources.

## What Happens When Threads Take Different Branches?

Threads may encounter conditional statements such as:

```c
if (value > 0) {
    result = value * 2;
} else {
    result = 0;
}
```

If all threads in a warp choose the same branch, the warp executes efficiently.

However, if some threads choose the `if` branch while others choose the `else` branch, the warp experiences branch divergence.

The GPU typically handles divergence by:

1. Executing one branch for the threads that selected it.
2. Disabling the other threads temporarily.
3. Executing the other branch for the remaining threads.
4. Combining the results after both paths finish.

For example:

```text
Warp:
Threads 0–15  → if branch
Threads 16–31 → else branch
```

The GPU executes both branches, but only the relevant threads are active during each one.

This reduces performance because the warp cannot execute both paths fully in parallel. Branch divergence is most costly when threads within the same warp frequently follow different control paths.

Branch divergence is less problematic when entire warps follow the same branch, even if different warps choose different branches.