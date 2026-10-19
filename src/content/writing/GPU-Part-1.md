---
title: "GPUs Part 1"
description: "Some idea on GPU."
pubDate: 2026-10-18
category: "AI"
tags:
  - NLP
  - AI
  - Transformers
thumbnail: ../../assets/writing/gpu.png
thumbnailAlt: "Description of the image"
---

# Part 1 — Why GPUs Exist

## Why Do We Need GPUs?

Modern applications process enormous amounts of data. A CPU can handle these tasks, but it is not always efficient when the same operation must be repeated across thousands or millions of data elements.

Graphics rendering, scientific simulations, video processing, and artificial intelligence all require large numbers of calculations. GPUs were created to perform these calculations more efficiently by processing many operations at the same time.

A GPU does not replace a CPU. The CPU manages general-purpose tasks and complex program logic, while the GPU accelerates workloads that can be divided into many smaller operations.

## CPU vs GPU: The Fundamental Difference

A CPU contains a small number of powerful cores. Each core is designed to execute complex instructions, handle decisions, and manage different types of tasks. This makes the CPU suitable for operating systems, applications, and sequential programs.

A GPU contains a large number of smaller processing units. These units are designed to perform similar operations on many pieces of data simultaneously.

The fundamental difference is their design goal:

- A CPU focuses on completing a small number of tasks quickly.
- A GPU focuses on completing a large number of similar tasks simultaneously.

For example, editing an image may require applying the same calculation to every pixel. A CPU can process these pixels, but a GPU can distribute the work across many processing units and handle numerous pixels at the same time.

## Parallel Computing: The Core Idea

Parallel computing divides a large problem into smaller problems that can be processed simultaneously.

Consider an image containing one million pixels. If every pixel needs the same color adjustment, the operation does not need to be performed strictly one pixel after another. The pixels can be divided among many processing units, with each unit handling a different group.

This approach reduces the time required to complete the operation.

Parallel computing is most effective when:

- The same operation is repeated many times.
- Individual data elements can be processed independently.
- The workload can be divided into smaller parts.
- The results can later be combined or used together.

Image processing, matrix multiplication, neural-network operations, and scientific simulations often contain this type of parallel structure.

## From Sequential Execution to Massive Parallelism

Traditional programs often follow a sequential execution model. Instructions are executed one after another, and each operation may depend on the result of a previous operation.

Sequential execution is useful for tasks involving complex logic, branching, and dependencies. However, it becomes inefficient when a program performs the same calculation on a large amount of independent data.

GPUs address this limitation by executing many threads in parallel. Each thread performs a small part of the overall task. When thousands of threads work together, the GPU can process large datasets efficiently.

For example, a neural network performs a huge number of multiplication and addition operations. Many of these operations are independent and can be executed simultaneously. This makes GPUs highly effective for deep learning and other data-intensive workloads.

The transition can be understood as:

- Sequential execution: one operation is processed after another.
- Multicore execution: several operations are processed simultaneously.
- GPU execution: thousands of lightweight operations are processed in parallel.

This shift from sequential execution to massive parallelism is the main reason GPUs are important in modern computing.