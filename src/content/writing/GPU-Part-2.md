---
title: "GPUs Part 2"
description: "Some idea on GPU."
pubDate: 2026-10-19
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/gpu.png
thumbnailAlt: "Description of the image"
---
# Part 2 — The GPU Programming Model

## CPU and GPU: Host vs Device

In GPU programming, the CPU and GPU have separate roles.

The CPU is called the **host**, while the GPU is called the **device**.

The host controls the application and manages tasks such as:

- Creating data.
- Allocating memory.
- Copying data between CPU and GPU memory.
- Launching GPU operations.
- Synchronizing with the GPU.

The device executes computationally intensive work using many parallel threads.

A typical GPU program follows this sequence:

1. The CPU prepares the input data.
2. The CPU transfers the data to GPU memory.
3. The CPU launches a function on the GPU.
4. The GPU processes the data in parallel.
5. The GPU stores the results in its memory.
6. The CPU retrieves the results.

The CPU and GPU usually have separate memory spaces, so data often needs to be transferred between them.

## What Is a Kernel?

A kernel is a function that runs on the GPU.

Unlike a regular CPU function, a kernel is executed by many GPU threads simultaneously. Each thread usually processes a different part of the input data.

For example, a vector-addition kernel may calculate one output element per thread:

```text
Thread 0 → output = input_a + input_b
Thread 1 → output = input_a + input_b
Thread 2 → output = input_a + input_b
```

The kernel contains the instructions, while the threads provide the parallel execution.

A kernel is suitable for GPU execution when the same operation must be applied to many independent data elements.

## What Happens When a Kernel Is Launched?

When the CPU launches a kernel, it specifies the number of threads that should execute it. The GPU then organizes these threads into a hierarchy of grids, blocks, warps, and individual threads.

The process generally works as follows:

1. The CPU calls the kernel.
2. The launch configuration defines the grid and block dimensions.
3. The GPU creates the requested logical threads.
4. The threads are grouped into blocks.
5. The blocks are distributed across the GPU’s streaming multiprocessors.
6. Each block is divided into warps.
7. The warps execute the kernel instructions.
8. The threads produce the output.

The CPU does not usually execute the kernel instructions itself. It only requests that the GPU execute them.

## The GPU Execution Hierarchy

GPU threads are organized into multiple levels:

```text
Grid
└── Blocks
    └── Warps
        └── Threads
```

Each level has a different purpose:

- The grid represents the complete kernel launch.
- A block is a group of cooperating threads.
- A warp is a hardware execution group.
- A thread performs an individual instance of the kernel.

This hierarchy allows the GPU to organize large numbers of parallel operations efficiently.

## Grid

A grid contains all the blocks created for a single kernel launch.

When a kernel is launched, the programmer defines the grid dimensions. The grid may be one-dimensional, two-dimensional, or three-dimensional, depending on the structure of the data.

Examples include:

- A one-dimensional grid for processing an array.
- A two-dimensional grid for processing an image.
- A three-dimensional grid for processing volumetric data.

Blocks within the same grid are generally designed to operate independently. This allows the GPU to schedule them across available processing units.

## Block

A block is a group of threads within a grid.

Threads in the same block can:

- Communicate through shared memory.
- Synchronize with one another.
- Cooperate on a common task.

A block is assigned to a streaming multiprocessor, where it remains until its execution is complete. The GPU can run multiple blocks on the same multiprocessor if enough resources are available.

Blocks are useful for dividing a large task into smaller independent sections.

For example, when processing an array, each block may handle a different range of elements:

```text
Block 0 → elements 0–255
Block 1 → elements 256–511
Block 2 → elements 512–767
```

## Warp

A warp is a group of threads scheduled and executed together by the GPU hardware.

In NVIDIA GPUs, a warp typically contains 32 threads. These threads execute the same instruction at the same time on different data.

This execution model is called SIMT, which means **Single Instruction, Multiple Threads**.

For efficient execution, threads within a warp should generally follow the same control path. If threads take different branches of a conditional statement, the GPU may execute those branches separately, reducing efficiency. This situation is known as branch divergence.

## Thread

A thread is the smallest logical unit of GPU execution.

Each thread executes the kernel independently and usually works on a specific data element. Threads have their own:

- Registers.
- Thread index.
- Program state.
- Local variables.

Threads can identify their position using built-in indices. These indices allow each thread to determine which part of the data it should process.

For example:

```text
Thread 0 processes data
Thread 1 processes data
Thread 2 processes data
```

Although threads execute the same kernel, their different indices allow them to operate on different data.

## Grid → Block → Warp → Thread: The Complete Picture

The complete GPU execution structure can be represented as:

```text
Grid
├── Block 0
│   ├── Warp 0
│   │   ├── Thread 0
│   │   ├── Thread 1
│   │   └── ...
│   └── Warp 1
│       ├── Thread 32
│       ├── Thread 33
│       └── ...
└── Block 1
    ├── Warp 0
    └── Warp 1
```

The relationships are:

- A grid contains multiple blocks.
- A block contains multiple threads.
- The hardware divides each block into warps.
- A warp contains multiple threads that execute together.
- Each thread executes one instance of the kernel.

For example, a kernel launch may contain:

```text
Grid: 4 blocks
Each block: 256 threads
Each warp: 32 threads
```

This produces:

```text
Total threads = 4 × 256 = 1,024
Warps per block = 256 ÷ 32 = 8
Total warps = 4 × 8 = 32
```

The programmer usually defines the grid and block sizes. The GPU hardware manages the scheduling of blocks and warps on its processing units.