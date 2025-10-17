# Efficiently Serving Large Language Models

This repository contains detailed notes and explanations for the concepts covered in the **[Efficiently Serving Large Language Models](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)** course by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[PrediBase](https://predibase.com/)**.

-----

## Introduction

Large Language Models (LLMs) have demonstrated incredible capabilities, but their sheer size presents a significant challenge for deployment. A model like Llama 3 70B requires over 140GB of VRAM just to store its weights in 16-bit precision, making inference slow and expensive. Running these models for a single user is costly; serving them to thousands of concurrent users efficiently is one of the most significant engineering challenges in modern AI.

This course, and by extension this repository, delves into the stack of optimizations that make it possible to serve LLMs at scale. We move from the naive, "one-at-a-time" autoregressive generation to a highly-optimized, multi-tenant serving architecture.

The key optimizations we will explore include:

  * **KV Caching:** The foundational optimization to prevent re-computation during text generation.
  * **Batching & Continuous Batching:** The single most important technique for increasing server throughput by maximizing GPU utilization.
  * **Quantization:** Techniques for shrinking model size (e.g., from FP16 to INT4) to fit larger models on smaller hardware and reduce memory bandwidth bottlenecks.
  * **Low-Rank Adaptation (LoRA):** A Parameter-Efficient Fine-Tuning (PEFT) method that allows for mass customization of LLMs.
  * **Multi-LoRA Inference (LoRAX):** The cutting-edge serving technique that allows a single server to host *thousands* of different LoRA-fine-tuned models at once.

This document serves as a deep dive into each of these topics, expanding on the core ideas presented in the course.

-----

## Course Topics

Here is a detailed breakdown of each core topic from the curriculum.

### 1. Text Generation

At its core, an LLM is an **autoregressive** model. This is a simple-sounding term for a deeply consequential process: the model generates text *one token at a time*, and each new token it generates is fed back into the model to help decide the *next* token. This sequential, step-by-step process is the fundamental reason LLM inference is slow and compute-intensive.

Understanding this loop is the first step to optimizing it. The entire field of LLM inference optimization is dedicated to making this one-token-at-a-time loop as fast and efficient as possible.

<details>
<summary><b>The Autoregressive Generation Loop</b></summary>

Let's break down the generation process step-by-step for a prompt like "The capital of France is".

1.  **Tokenization:** The input text is split into tokens. `["The", "capital", "of", "France", "is"]`.
2.  **Prompt Processing (Prefill Phase):** The model performs a *single, large forward pass* on these initial tokens. This pass is computationally expensive but only happens once. Its primary goal is not to generate text, but to "warm up" the model's internal state.
3.  **The "State" (KV Cache):** The "internal state" of a Transformer is its **Key-Value (KV) Cache**. In the self-attention mechanism, each token creates a Query (Q), Key (K), and Value (V) vector. The K and V vectors represent that token's "identity" and "content." For the model to generate the *next* token, it must be able to "see" (attend to) all the K and V vectors of the *previous* tokens. Instead of recomputing them every single time, we store them in a cache on the GPU. This is the **KV Cache**, and it is the *most important baseline optimization* for LLM inference. Without it, inference would be unusable.
4.  **Generation Phase (Decoding Loop):** This is the one-token-at-a-time loop.
      * **Step 1:** The model takes the KV Cache from the prompt and performs a *small* forward pass using *only* the last token's information to produce *logits*—a massive vector of probabilities for every possible next token in its vocabulary.
      * **Step 2 (Sampling):** A sampling algorithm (e.g., greedy search, top-k, top-p) picks *one* token from these logits. Let's say it picks "Paris".
      * **Step 3 (Append):** The new token "Paris" is fed back into the model.
      * **Step 4 (Update Cache):** The model performs *another* forward pass, calculating the K and V vectors for *only* the token "Paris" and *appends* them to the KV Cache. The cache now holds the K/V pairs for `["The", "capital", "of", "France", "is", "Paris"]`.
      * **Step 5 (Repeat):** The model repeats this process, using the *entire* updated cache to generate the *next* token (e.g., "."), then the next (e.g., `[EOS]` or End-of-Sequence token).
5.  **Termination:** The loop stops when the model generates a special `[EOS]` token or it reaches a pre-defined maximum length.

The problem is that this generation loop is bottlenecked by **memory bandwidth**. For *every single token*, the GPU must read the *entire* model's weights (e.g., 140GB for Llama 70B) from its VRAM. This I/O operation is often slower than the actual math (the FLOPs), meaning the expensive compute cores of the GPU are sitting idle, waiting for data. Furthermore, the KV Cache itself grows with every token, consuming a massive amount of VRAM. A 10-user batch with a 2048-token context window can create a KV Cache that is *larger* than the model weights themselves. These two problems—the memory bandwidth bottleneck and the VRAM cost of the KV Cache—are what drive the need for all subsequent optimizations.

</details>

### 2. Batching

The simplest way to serve an LLM is to process one user request at a time. However, as established, a single request (batch size 1) is extremely inefficient and cannot "saturate" a powerful GPU. The GPU's parallel processors are left with nothing to do. The obvious solution, borrowed from classic machine learning, is **batching**: grouping multiple user requests together and processing them as a single, large batch.

This increases **throughput** (total tokens processed per second) by keeping the GPU busy. However, the *autoregressive* nature of LLMs makes this "static" batching deeply problematic and inefficient.

<details>
<summary><b>The Failures of Static Batching</b></summary>

**Static Batching** (or "naive batching") works like this:

1.  A server collects requests in a queue (e.g., `req_1`, `req_2`, `req_3`).
2.  It waits until it has a full batch (or a timeout is hit).
3.  It finds the *longest* prompt in the batch (e.g., `req_2` has 100 tokens).
4.  It **pads** all *other* prompts with `[PAD]` tokens so they are all 100 tokens long.
5.  It stacks these prompts into a single tensor and runs the *prefill phase* (see Topic 1).
6.  It then enters the *generation phase*, generating one token for every request in the batch at the same time.

This seems efficient, but it has two fatal flaws:

**Flaw 1: Head-of-Line Blocking & Padding Waste**
The entire batch is now treated as a single unit. This means the batch *cannot finish* until the *longest generation* is complete.

  * Imagine `req_1` ("What is 2+2?") finishes in 3 tokens: "4. `[EOS]`"
  * Imagine `req_2` ("Write me a poem about a...") needs to generate 500 tokens.

Even though `req_1` finished almost instantly, its "slot" in the batch remains active. The server *must* continue to run computations for it—useless `[PAD]` tokens—for 497 more steps until `req_2` finally finishes.

This is a *catastrophic* waste of GPU compute. This "internal fragmentation" means that as the batch progresses, more and more of its slots become idle, and GPU utilization plummets. It also means the latency for `req_1` is no longer 1 second; its latency is now tied to `req_2`'s 2-minute generation. This is unacceptable for a production system.

**Flaw 2: Inflexible Scheduling**
With static batching, the server cannot add *new* requests to a batch that is already in progress. It must wait for the *entire* batch to complete (i.e., for `req_2` to finish its 500 tokens) before it can start processing `req_4`, `req_5`, and `req_6`, which have been waiting in the queue.

This leads to high wait times for users and "spiky" GPU usage: the GPU is 100% busy for two minutes, then briefly idle as it swaps in the next batch. We need a way to *decouple* the batch from the lifecycle of individual requests. This is the motivation for Continuous Batching.

</details>

### 3. Continuous Batching

Continuous Batching (also known as "dynamic batching") is the solution to the failures of static batching. It is arguably the **single most important optimization for LLM throughput**. The core idea is simple: instead of batching *requests*, we batch *iterations*. The server maintains a "running batch" of active requests, and on *every single step*, it does two things:

1.  Generates one token for all requests *currently* in the batch.
2.  *Removes* any requests that just generated an `[EOS]` token.
3.  *Adds* new requests from the queue into the newly-freed slots.

This way, the batch is *always* full, and the GPU is *always* operating at maximum capacity. A request that takes 3 tokens (`req_1`) will be in the batch for 3 steps and then leave. A request that takes 500 tokens (`req_2`) will be in the batch for 500 steps. They can co-exist, and neither has to wait for the other.

This simple idea dramatically increases throughput (often by 10-20x) and GPU utilization, but it introduces a major engineering challenge: memory management.

<details>
<summary><b>The Memory Problem: PagedAttention</b></summary>

In the world of continuous batching, we have a batch of requests (`req_1`, `req_2`, `req_3`...) all at *different stages* of generation. This means their **KV Caches** (from Topic 1) are all *different sizes*.

  * `req_1` might have a 50-token KV Cache.
  * `req_2` might have a 1500-token KV Cache.
  * `req_3` (a new request) might have a 10-token KV Cache.

How do you store these different-sized, non-contiguous, dynamically-growing blocks of data in GPU memory?

**The Naive Solution (Bad):** Pre-allocate the *maximum* context length (e.g., 4096 tokens) for *every* request. If `req_1` only uses 50 tokens, the other 4046 tokens' worth of VRAM is wasted. This "internal fragmentation" is so severe that you'd only be able to fit a few requests in memory, defeating the purpose of batching.

**The Smart Solution (PagedAttention):** This is the key innovation from the vLLM project, which is now widely adopted. It borrows a core concept from operating systems: **virtual memory** and **paging**.

1.  **Blocks:** Instead of allocating one giant, contiguous chunk of memory for each request, PagedAttention *divides the entire KV cache space* on the GPU into thousands of small, fixed-size **"blocks"** (e.g., 16 tokens each).
2.  **Block Tables:** Each request is given a "block table," which is like a virtual-to-physical page table in an OS. This table maps the *logical* token index (e.g., "token \#52") to the *physical* block where its K/V data is *actually* stored.
3.  **Dynamic Allocation:**
      * When a request starts, it gets a few blocks.
      * When it generates a new token and needs *more* cache space, the PagedAttention kernel simply *allocates one new block* from the GPU's free block pool and *adds its address to the request's block table*.
      * When a request *finishes*, its blocks are *instantly returned* to the free pool, ready for a new request to use.

**Why this is revolutionary:**

  * **No Internal Fragmentation:** A request with 50 tokens uses *exactly* the blocks it needs (e.g., 4 blocks). A request with 1500 tokens uses *its* required blocks. There is *zero* wasted space from pre-allocation.
  * **No External Fragmentation:** Because blocks are small and fixed-size, the memory space doesn't get "fragmented" with weirdly-sized holes.
  * **Efficient Sharing:** PagedAttention even allows for complex sharing. For example, in *parallel sampling* (where you generate 3 different outputs for the *same* prompt), the 3 requests can *all share the same blocks* for the prompt phase, and only allocate new, separate blocks for their *different* generated tokens.

**Conclusion:** Continuous Batching is the *scheduling algorithm* that maximizes throughput. PagedAttention is the *memory management system* that makes continuous batching feasible and hyper-efficient. Together, they form the foundation of all modern, high-performance LLM inference servers like vLLM, TGI, and PrediBase's LoRAX.

</details>

### 4. Quantization

We've optimized scheduling (Continuous Batching) and memory management (PagedAttention), but we still have two fundamental problems:

1.  **Model Size:** A 70B model (140GB at FP16) doesn't fit on an 80GB A100 GPU.
2.  **Memory Bandwidth:** Even if it did, the GPU is *still* bottlenecked by reading those 140GB of weights from VRAM for *every single token*.

**Quantization** is the solution. It is the process of *reducing the precision* (the number of bits) used to store the model's weights. Instead of storing a weight as a 16-bit floating-point number (FP16), we store it as an 8-bit integer (INT8) or even a 4-bit integer (INT4).

This has two *massive* benefits:

1.  **Reduces VRAM Footprint:**
      * FP16 (16-bit): 70B model = 140 GB
      * INT8 (8-bit): 70B model = 70 GB (Now fits on an A100\!)
      * INT4 (4-bit): 70B model = 35 GB (Now fits on a consumer 4090\!)
2.  **Increases Speed:** This is the most critical part. The memory bandwidth bottleneck is *instantly* reduced. Loading 4-bit weights from VRAM is *4x faster* than loading 16-bit weights. The GPU spends less time *waiting* for data and more time *computing*, leading to a significant increase in token generation speed.

<details>
<summary><b>Types and Trade-offs of Quantization</b></summary>

Quantization is not "free." By using fewer bits, you are *losing information* and *introducing error*. A weight that was `3.14159` in FP16 might become `3.1` in INT8. The goal of a good quantization algorithm is to minimize the *impact* of this error on the model's output quality.

**Data Types:**

  * **FP32 (32-bit):** The "gold standard" for training. Not used for inference.
  * **FP16 / BF16 (16-bit):** The standard "full precision" for inference. This is our baseline.
  * **INT8 (8-bit):** The most common and well-supported quantization. It cuts memory in half and provides a good speedup with a *very small* accuracy loss (if done correctly).
  * **INT4 / NF4 (4-bit):** A more "aggressive" quantization. It cuts memory by 4x. This is what projects like `GPTQ` and `bitsandbytes` (NF4, or NormalFloat4) use. The accuracy loss is larger than INT8, but often still negligible for many tasks.

**Quantization Techniques:**

1.  **Post-Training Quantization (PTQ):** This is the most common and popular method. You take a *fully-trained* model and *then* apply a quantization algorithm to it.

      * **Naive (Min/Max):** The simplest way. Find the *minimum* and *maximum* weight values in a tensor (e.g., `-10.5` and `+12.2`). Map this range to the INT8 range (`-127` to `+127`). This works poorly, as a *single outlier* (e.g., one weight at `+1000.0`) will *destroy* the precision for all other weights, which might be clustered between `-1` and `+1`.
      * **GPTQ (Generative Pre-trained Transformer Quantization):** A "one-shot" method. It quantizes the model layer by layer. As it quantizes one weight, it *adjusts the remaining, un-quantized weights* in that layer to *compensate* for the error it just introduced. This results in a *much* more accurate quantized model.
      * **AWQ (Activation-aware Weight Quantization):** A very clever, recent technique. It observes that not all weights are created equal. Some weights are *more important* than others because they are consistently multiplied by *large activation values* during inference. AWQ's insight is to *protect* these 1% "important" weights, keeping them in higher precision, while *aggressively quantizing* the other 99% of "unimportant" weights. This yields state-of-the-art accuracy for 4-bit quantization.

2.  **Quantization-Aware Training (QAT):** A more complex but powerful method. You *simulate* the quantization *during* the fine-tuning process. The model "learns" to work around the precision loss. This gives the best possible accuracy but requires a full (and expensive) fine-tuning run, so it's less common than PTQ.

By combining Quantization (to fit the model and speed up I/O) with Continuous Batching (to keep the GPU busy), we can build a *very* fast server for a *single* base model. But what if we have 1,000 different *customers* who all want their *own* fine-tuned version?

</details>

### 5. Low-Rank Adaptation (LoRA)

This is where the paradigm shifts from *serving* to *customization*. Full fine-tuning (FFT) of an LLM—retraining all 70 billion parameters on a new task—is *prohibitively* expensive. It creates an entirely new 140GB model file and requires a massive amount of VRAM to train.

**LoRA (Low-Rank Adaptation)** is a **Parameter-Efficient Fine-Tuning (PEFT)** technique that solves this. It's based on a simple but powerful hypothesis: when you fine-tune a model, the *change* in the weights (the "delta") doesn't need to modify all 70B parameters. The *change* itself is "low-rank," meaning it can be represented with far fewer parameters.

Instead of modifying the *original* weights $W$ (which are, say, $4096 \times 4096$), LoRA *freezes* them. It then injects a *new, tiny "adapter"* path next to it. This path consists of two *much smaller* matrices, $A$ (e.g., $4096 \times 8$) and $B$ (e.g., $8 \times 4096$).

During fine-tuning, only the $A$ and $B$ matrices are trained.

<details>
<summary><b>The Math and Impact of LoRA</b></summary>

**The Math (Forward Pass):**

  * Original model's output: $h = W \cdot x$
  * LoRA model's output: $h = (W \cdot x) + (B \cdot A \cdot x)$

**The Parameter Savings (The "Why"):**

  * Parameters in $W$: $4096 \times 4096 = 16,777,216$ (all frozen)
  * Parameters in $A$ and $B$ (with $r=8$): $(4096 \times 8) + (8 \times 4096) = 32,768 + 32,768 = 65,536$ (all trainable)
  * **Savings:** We are training $65,536$ parameters instead of $16,777,216$. This is a **\~256x reduction** in trainable parameters.

This has *revolutionary* implications:

1.  **Fast, Cheap Training:** Fine-tuning is *dramatically* faster and can be done on consumer GPUs, as the optimizer only needs to store gradients for the tiny $A$ and $B$ matrices.
2.  **No "Catastrophic Forgetting":** Because the original 70B weights are *frozen*, the model *cannot* "forget" its original knowledge. It's just *learning a new adaptation* on top.
3.  **Massive Storage Savings:** This is the key. Instead of saving 1,000 *different* 140GB models (140 TB total), a company can now store:
      * **One** 140GB base model.
      * **1,000** tiny LoRA adapters (e.g., 10-100MB each).
        This reduces the storage footprint from *terabytes* to *gigabytes*.

**The Deployment Problem:**
After training, you have a `base_model` and a `lora_adapter.safetensors` file. For deployment, you *could* "merge" them by literally computing $W' = W + (B \cdot A)$ and saving the new 140GB model. This gives you a new, standalone model with *zero* inference overhead.

**But...** what if you have 1,000 adapters? If you merge them, you're back to 1,000 full-sized models, defeating the storage savings. If you *don't* merge them, how can you *serve* them? You can't just swap them in and out of VRAM for every request; the I/O latency would be terrible.

This creates a new $100M challenge: How do you serve *thousands* of *different* LoRA adapters *concurrently* on the *same GPU*, all sharing the *same* base model?

</details>

### 6. Multi-LoRA Inference

**Multi-LoRA Inference** is the technique for solving the problem LoRA created. The goal is to take a batch of user requests, where *each request needs a different LoRA adapter*, and process them all *at the same time* in a *single forward pass*.

This is *extremely* difficult. Let's look at a batch of 3 requests:

  * `req_1` (User A): needs `lora_adapter_A`
  * `req_2` (User B): needs `lora_adapter_B`
  * `req_3` (User C): also needs `lora_adapter_A`

In a single forward pass, the GPU must compute:

  * $h_1 = (W \cdot x_1) + (B_A \cdot A_A \cdot x_1)$
  * $h_2 = (W \cdot x_2) + (B_B \cdot A_B \cdot x_2)$
  * $h_3 = (W \cdot x_3) + (B_A \cdot A_A \cdot x_3)$

The $W \cdot x$ part is easy; that's a standard batched matrix multiply (BMM). The hard part is the LoRA delta: $(B_i \cdot A_i \cdot x_i)$. This calculation is *different* for *every request* in the batch. It requires new, highly-specialized CUDA kernels that can perform these "grouped" matrix-vector multiplies, where each item in the batch is multiplied by a *different* set of adapter weights.

<details>
<summary><b>The "S-LoRA" and "LoRAX" Approaches</b></summary>

Two primary approaches emerged to solve this: S-LoRA and LoRAX. The PrediBase LoRAX framework is a production-grade implementation of this concept.

The core idea is to *keep the adapters separate* (un-merged) in VRAM.

1.  **Base Model Pass:** The server first runs the $W \cdot x$ operation for *all* requests in the batch. This is the main BMM and uses the shared, frozen base model weights.
2.  **Adapter Weight Gathering:** The server identifies *which* adapters are needed for the *current* batch (in our example, `lora_A` and `lora_B`). It pulls these (very small) $A$ and $B$ matrices from its VRAM cache.
3.  **LoRA Delta Pass:** The server uses a highly optimized, custom kernel to perform the parallel LoRA computations. This kernel is smart. It knows that `req_1` and `req_3` *share* `lora_A`, so it can group their computations. It computes the "delta" (the `+ (B \cdot A \cdot x)` part) for *every request* in the batch *simultaneously*.
4.  **Aggregation:** The server simply *adds* the output of the base pass and the LoRA delta pass to get the final result for each request.

**The Memory Problem (Again):**
This is brilliant, but what if you have *one million* adapters? Even if they are 50MB each, they can't all fit in VRAM.

This is where the final piece of the puzzle comes in: **Dynamic Adapter Loading** and **Tiered Caching**.

A server like LORAX (see next topic) doesn't *just* hold adapters in VRAM. It creates a *cache hierarchy*:

  * **Tier 1: GPU VRAM (Hot):** The adapters used in the last few seconds (e.g., `lora_A` and `lora_B`). Access is instantaneous.
  * **Tier 2: CPU RAM (Warm):** Adapters that were used recently (e.g., in the last 10 minutes).
  * **Tier 3: Disk / S3 (Cold):** The other 999,990 adapters.

When a new request for `lora_C` (which is "warm") comes in, the server *in the background* initiates a high-speed *CPU-to-GPU transfer*. When `lora_C` arrives in VRAM, it's added to the *next* batch. If the VRAM cache is full, the server *evicts* the "least recently used" (LRU) adapter back to CPU RAM to make space.

This system allows the server to *transparently* handle a virtually *unlimited* number of adapters, while maintaining near-native performance for the most popular ones. It combines the *throughput* of Continuous Batching with the *mass-customization* of LoRA.

</details>

### 7. LORAX

**LORAX (LoRA eXchange)** is the open-source production inference server from PrediBase that implements *all* of the concepts discussed so far into a single, cohesive system.

It is the "capstone" that brings everything together. It is *not* just a "multi-LoRA" server; it is a full-stack, high-performance LLM server that has *native support* for Multi-LoRA inference.

Let's review how LORAX combines all 7 topics:

1.  **Text Generation:** It is built from the ground up to handle the **autoregressive** nature of LLMs, including managing the **KV Cache**.
2.  **Batching:** It *rejects* static batching.
3.  **Continuous Batching:** It implements **continuous batching** as its core scheduling strategy to ensure the GPU is always saturated, maximizing throughput. It likely uses a **PagedAttention**-like memory manager to handle the KV Caches of all the requests in its dynamic batch.
4.  **Quantization:** LORAX allows the *base model* $W$ to be loaded in a **quantized** format (e.g., INT4 or INT8 via GPTQ/AWQ). This *dramatically* reduces the VRAM footprint and speeds up the "base model pass," allowing you to serve 1,000 adapters on a 70B model using a single, affordable GPU.
5.  **LoRA:** It is *built for* LoRA. It is designed around the "un-merged" adapter paradigm, where adapters are treated as separate, pluggable entities.
6.  **Multi-LoRA Inference:** LORAX implements the custom CUDA kernels (like S-LoRA's) needed to *batch* requests for *different* LoRA adapters *at the same time*.
7.  **LORAX (The Framework):** It adds the final, production-ready layer: the **Dynamic Adapter Loading** and **Tiered Caching** (VRAM -\> CPU RAM -\> Disk) system. This is what makes it "infinitely" scalable, allowing a single server to handle *millions* of adapters, intelligently swapping the hot ones onto the GPU as needed.

<details>
<summary><b>Putting It All Together: A LORAX Request Lifecycle</b></summary>

1.  A *batch* of 10 requests arrives at the LORAX server.
      * 5 requests are for `lora_sales_bot` (in VRAM).
      * 3 requests are for `lora_support_bot` (in VRAM).
      * 2 requests are for `lora_legal_doc` (in CPU RAM).
2.  LORAX's scheduler sees this. It *immediately* builds a new batch with the 8 requests for the "hot" adapters.
3.  *In the background*, it triggers a DMA transfer of `lora_legal_doc` from CPU to GPU VRAM. It *evicts* the least-recently-used adapter from VRAM to make room.
4.  The server's **quantized** (Topic 4) base model $W$ is on the GPU.
5.  The **Continuous Batching** (Topic 3) engine runs a forward pass:
      * It runs the "base pass" $W \cdot x$ for all 8 requests on the quantized weights.
      * It uses its **Multi-LoRA** (Topic 6) kernel to compute the `(B \cdot A \cdot x)` deltas, using the `lora_sales_bot` weights for 5 requests and `lora_support_bot` weights for 3.
      * It adds the results and generates *one* new token for all 8 requests.
6.  One of the `lora_sales_bot` requests generates `[EOS]`.
7.  By now, `lora_legal_doc` has arrived in VRAM.
8.  In the *very next* iteration, the **Continuous Batching** scheduler *removes* the finished request and *adds* the 2 `lora_legal_doc` requests (plus a new request from the queue) to the batch.
9.  The process repeats. The GPU is *always* 100% busy, the batch is *always* full, and adapters are *dynamically* paged in and out of VRAM, allowing the server to handle *thousands* of customers concurrently on a *single* piece of hardware.

This is the state-of-the-art in efficient, multi-tenant LLM serving.

</details>

-----

## Acknowledgement

This repository is for educational and learning purposes only. The content and concepts summarized here are from the **[Efficiently Serving Large Language Models](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)** course.

All rights, course materials, and trademarks are held by **[DeepLearning.AI](https://learn.deeplearning.ai/)** and **[PrediBase](https://predibase.com/)**.
