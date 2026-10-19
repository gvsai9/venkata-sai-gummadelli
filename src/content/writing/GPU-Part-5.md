---
title: "GPUs Part 5"
description: "Some idea on GPU."
pubDate: 2026-10-22
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/gpu.png
thumbnailAlt: "Description of the image"
---
# Part 5 — GPU Memory Architecture

## Why GPU Memory Is So Important

GPU performance depends not only on how quickly calculations are performed, but also on how quickly data can be accessed.

A GPU may contain thousands of processing threads. If these threads wait for data from memory, the computational units remain idle.

GPU memory is organized into different levels. Each level has different characteristics:

- Capacity.
- Access speed.
- Location.
- Visibility.
- Lifetime.
- Cost of access.

Efficient GPU programs keep frequently used data in faster memory and access global memory in an organized way.

## The GPU Memory Hierarchy

The main levels of GPU memory are:

```text
Registers
   ↓
Shared Memory and L1 Cache
   ↓
L2 Cache
   ↓
Global Memory
```

Registers are the fastest memory available to a thread but have very limited capacity.

Shared memory is fast memory shared by threads within the same block.

Global memory has the largest capacity but usually requires more time to access.

The hierarchy exists because fast memory is expensive and limited, while large memory is slower and more abundant.

## Global Memory

Global memory is the main memory space available to GPU programs.

It is accessible by:

- All threads.
- All blocks.
- The CPU through memory-transfer operations.

Global memory usually has large capacity, making it suitable for storing:

- Input data.
- Output data.
- Model parameters.
- Large arrays.
- Images.
- Matrices.

However, global-memory access is slower than access to registers or shared memory. Efficient programs try to reduce unnecessary global-memory operations.

Global memory is also used to transfer data between the CPU and GPU. These transfers can become a performance bottleneck if they occur too frequently.

## Shared Memory

Shared memory is a fast memory space located on an SM.

It is shared by all threads within the same block. Threads in different blocks cannot directly use each other’s shared memory.

Shared memory is useful when multiple threads need to reuse the same data.

For example, in matrix multiplication, threads in a block can load a section of a matrix into shared memory. They can then reuse that data instead of repeatedly loading it from global memory.

Shared memory has:

- Low access latency.
- Limited capacity.
- Block-level visibility.
- A lifetime limited to the execution of the block.

Threads often use synchronization to ensure that shared data has been loaded before other threads access it.

## Registers

Registers are the fastest memory available to a thread.

Each thread has its own private registers. Registers commonly store:

- Temporary calculations.
- Local variables.
- Thread-specific values.
- Frequently used data.

Other threads cannot directly access a thread’s registers.

The number of registers available to a thread is limited. If a kernel uses too many registers, some data may be moved to local memory, which is significantly slower.

High register usage can also reduce the number of active threads that an SM can support.

## Local Memory

Local memory is private to an individual thread, but it is physically stored in global memory.

It may be used when:

- A thread requires more registers than are available.
- Local arrays cannot fit in registers.
- Certain variables cannot be stored in registers.

Although local memory is private in terms of access, it does not have the same speed as registers. Accessing it can introduce global-memory-level latency.

The name “local memory” describes visibility, not physical location.

## Constant Memory

Constant memory is a read-only memory space intended for values that remain unchanged during kernel execution.

It is useful for data such as:

- Configuration values.
- Mathematical constants.
- Small lookup tables.
- Parameters shared by many threads.

Constant memory can be efficient when many threads access the same address. The hardware can provide the same value to multiple threads efficiently.

It is less suitable when threads access many different constant-memory addresses at the same time.

## L1 and L2 Cache

Caches store recently used data so that future accesses can be served more quickly.

### L1 Cache

L1 cache is located close to the processing units on an SM. It is smaller and faster than L2 cache.

Depending on the GPU architecture and configuration, L1 cache may share hardware resources with shared memory.

### L2 Cache

L2 cache is shared across the GPU. It is larger than L1 cache but usually slower to access.

L2 cache can serve data requested by different SMs and helps reduce repeated accesses to global memory.

The cache hierarchy is managed largely by hardware. Programmers influence cache behavior indirectly through data layout and memory-access patterns.

## Who Can Access Which Memory?

Different memory spaces have different visibility rules.

| Memory | Accessed by | Main characteristic |
|---|---|---|
| Registers | One thread | Private and very fast |
| Local memory | One thread | Private but stored in global memory |
| Shared memory | Threads in the same block | Fast and cooperative |
| Global memory | All threads and the CPU through transfers | Large but slower |
| Constant memory | All threads for read-only access | Efficient for shared constant values |
| L1 cache | Hardware-managed | Close to an SM |
| L2 cache | Hardware-managed and shared across the GPU | Shared cache for global accesses |

The correct memory space depends on how the data is used.

Data used by only one thread should remain private. Data reused by threads in the same block may benefit from shared memory. Large input and output datasets generally reside in global memory.

## Memory Access Patterns

The way threads access memory has a major effect on performance.

### Coalesced Access

Coalesced access occurs when neighboring threads access neighboring memory locations.

```text
Thread 0 → data
Thread 1 → data
Thread 2 → data
Thread 3 → data
```

The GPU can combine these accesses into efficient memory transactions.

### Strided Access

Strided access occurs when threads access memory locations separated by a fixed distance.

```text
Thread 0 → data
Thread 1 → data
Thread 2 → data
Thread 3 → data
```

Large strides can require more memory transactions and reduce performance.

### Scattered Access

Scattered access occurs when threads access unrelated memory locations.

```text
Thread 0 → data
Thread 1 → data
Thread 2 → data
Thread 3 → data
```

This pattern generally uses memory bandwidth inefficiently.

Efficient GPU programs try to:

- Access neighboring memory locations with neighboring threads.
- Reuse data through shared memory or caches.
- Reduce unnecessary transfers between the CPU and GPU.
- Avoid excessive local-memory usage.
- Organize data according to the access pattern of the kernel.