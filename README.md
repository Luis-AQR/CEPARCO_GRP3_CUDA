# CEPARCO_GRP3_CUDA
## Members:
* Alcasid, Aizy
* Bilanes, Kendrick
* Corpuz, Gerald
* Roa, Luis

# Deep Dive: CUDA Concepts and Programming
## Project Specifications
**Kernel:** Vector AXPY on vectors $A$ and $B$, placing the result in vector $Z$.

**Input:** 
* Scalar variable $n$ (integer) contains the length of the vector.
* Scalar variable $a$ (single-precision float).
* Vectors $A$ and $B$ are single-precision floats.

**Process:**
$Z_i = a \cdot X_i + Y_i$ for $i = 0, 1, \dots, n-1$

**Output:** 
Store the result in vector $Z$. Display the execution time, the first 5, and the last 5 elements of vector $Z$ for verification across all kernels.

**Variants to be implemented:**
1. C/C++ program version
2. CUDA program version using a grid-stride loop without prefetch and without mem advise
3. CUDA program version using a grid-stride loop with prefetching but without page creation and without mem advise
4. CUDA program version using a grid-stride loop with prefetch, with page creation but without mem advise
5. CUDA program version using a grid-stride loop with prefetch, with page creation and with mem advise
   
## AI Usage Declaration

Gemini - Debug and check outputs of code
Grammarly - Spelling and Grammar checks, used in the README.md file

---

## I. Program Output and Correctness Check

**Variant 1: C/C++ Version**

![](images/ver1out.png)

**Variant 2: No Prefetch, No Mem Advise**

![](images/ver2out.png)

**Variant 3: With Prefetch, No Page Creation, No Mem Advise**

![](images/ver3out.png)

**Variant 4: With Prefetch, With Page Creation, No Mem Advise**

![](images/ver4out.png)

**Variant 5: With Prefetch, With Page Creation, With Mem Advise**

![](images/ver5out.png)

## II. NSYS Profiling

**Variant 2: No Prefetch, No Mem Advise**

![](images/ver2nsys.png)

**Variant 3: With Prefetch, No Page Creation, No Mem Advise**

![](images/ver3nsys.png)

**Variant 4: With Prefetch, With Page Creation, No Mem Advise**

![](images/ver4nsys.png)

**Variant 5: With Prefetch, With Page Creation, With Mem Advise**

![](images/ver5nsys.png)

---

## III. Comparative Table of Execution Time

| 2^24 elements | | |
| :--- | :--- | :--- |
| **CUDA block size = [256 \| 512 \| 1024]** | **Kernel time [avg \| min \| max] in ms** | **Speedup vs baseline C program** |
| C | [23.780529 \| 19.297711 \| 28.231964] | N/A |
| x86-64 | avg: 9.132447 | 2.60 |
| x86-64 SIMD XMM | avg: 7.635550 | 3.11 |
| x86-64 SIMD YMM | avg: 7.358211 | 3.23 |
| CUDA Unified (No Prefetch/Advise) | 256: [0.838689 \| 0.831475 \| 57.134721]<br>512: [0.868808 \| 0.825134  \| 1.834849]<br>1024: [0.838875 \| 0.819731 \| 0.969434] | 256: 28.35<br>512: 27.37<br>1024: 28.35 | |
| CUDA Prefetch (No Page Creation) | 256: [1.088286 \| 0.772256 \| 10.178272]<br>512: [1.076513 \| 0.778560 \| 9.653792]<br>1024: [1.053998 \| 0.782464 \| 8.875424] | 256: 21.85<br>512: 22.09<br> 1024: 22.56 |
| CUDA Prefetch + Page Creation | 256: [0.770405 \| 0.768544 \| 0.772672]<br>512: [0.775305 \| 0.772448 \| 0.784384]<br>1024: [0.782987 \| 0.780864 \| 0.791168] | 256: 30.87<br>512: 30.67<br>1024: 30.37 |
| CUDA Prefetch + Page Creation + Mem Advise| 256: [0.768346 \|0.766688 \| 0.770048]<br>512: [0.782662 \| 0.776192 \| 0.789248]<br>1024: [0.780159 \| 0.778240 \| 0.783872] | 256: 30.78<br>512: 30.22<br>1024: 30.31 |

---

## IV. Analysis of Results

### 1. Analysis of Kernel Execution Time
Across the evaluated implementations processing 2^24 elements, execution times strictly demonstrate the performance advantages of GPU execution and data prefetching over baseline CPU processing. The standard C baseline program posted an average execution time of 23.78 ms (with a minimum of 19.30 ms and a maximum of 28.23 ms).   

When moving to GPU variants, execution time drops dramatically. For the basic CUDA Unified Memory implementation, warm execution runs average between 0.838 ms and 0.868 ms depending on block size. However, this variant exhibits a severe maximum execution spike up to 57.13 ms (and up to 101.8 ms as noted in memory analysis) on its first run due to page migration overhead triggered on-demand by GPU page faults.   

In contrast, enabling prefetching completely eliminates this initial latency penalty by migrating pages to GPU VRAM prior to launch. Across block sizes 256, 512, and 1024, the prefetch variants maintain highly consistent execution times ranging from an average of 0.768 ms to 0.783 ms, with maximum times capped below 0.792 ms. Adding explicit memory hints, the cudaMemAdvise produces virtually identical performance (0.768 ms to 0.782 ms average), confirming that while memory advice provides explicit hints to the unified memory driver, prefetching alone is the main driver in achieving optimal kernel performance.   

### 2. Analysis of Speed Performance Across All Platforms
Upon comparing speedups relative to the baseline C program, GPU execution using SIMT substantially outperforms all CPU platforms. On the CPU side, scalar x86-64 execution provides a 2.60x speedup, while SIMD vectorization yields incremental gains—achieving 3.11x speedup with SIMD XMM and 3.23x speedup with SIMD YMM.   

The CUDA GPU implementations demonstrate a better overall throughput, achieving speedups ranging from 27.37x to 30.87x over baseline C. The highest overall performance is achieved by the CUDA Prefetch, with the Page Creation variant at a block size of 256, delivering a peak of 30.87x speedup, closely matched by the variant with cudaMemAdvise at 30.78x speedup.   

The CUDA/SIMT platform performed the best because AXPY is a parallel, memory-bandwidth-bound workload with zero data dependencies between vector elements. While CPU SIMD registers can process 4 to 8 float elements per instruction on a handful of cores, the GPU harnesses thousands of parallel CUDA threads executing across multiple Streaming Multiprocessors. When combined with prefetching, the GPU avoids host-device page faulting delays and fully leverages its high DRAM bandwidth to complete the 16.77 million element vector operations in less than one millisecond.  

### 3. Impact of Block Size
**Question:** How does block size affect execution time (observing various elements and using max blocks)? Which block size will you recommend?

Block size had only a small effect on kernel time, and smaller blocks were slightly faster. We recommend 256 threads per block.

For 2^24 elements (CUDA Prefetch + Page Creation, avg kernel time):
| **Block size** | **Avg (ms)** |
|---|---|
| 256 | 0.770 |
| 512 | 0.775 |
| 1024 | 0.783 |

Moving from 256 to 1024 threads per block increased the average time by only about 1.7%. Every launch used the maximum number of blocks for its size, `numBlocks = (N + numThreads - 1) / numThreads`, so bigger blocks mean fewer blocks. The T4 (compute capability 7.5) allows at most 1024 resident threads and 16 blocks per Streaming Multiprocessor (SM). With 1024 threads per block, only 1 block fits per SM. When that block finishes or stalls, there is little else to schedule.With 256 threads per block, 4 blocks fit per SM. The scheduler has more independent warps to hide memory latency and fill the SM.

**Recommendation**: 256 threads per block. It was marginally fastest, gives 4 resident blocks per SM on the T4, and is not tied to the 1024-thread limit. The grid-stride loop keeps the kernel correct whatever block size or block count is used.

### 4. Memory Management: Prefetching vs. CUDA Managed
**Question:** Is prefetching always recommended, or should CUDA manage memory? Give some use cases in which one is better than the other.

Prefetching is not always recommended, but for our AXPY kernel it is the better choice.

With plain `cudaMallocManaged`, pages migrate on demand. When the kernel touches a page that is still on the CPU, the GPU stalls those threads while the Page Migration Engine moves the page. That migration cost lands inside the kernel time. In our 2^24 results, the CUDA Unified variant had a maximum kernel time of 101.798 ms. Its minimum was 0.831 ms, so the first launch was about 120 times slower than steady state. With prefetching, the maximum dropped to about 0.78 ms. `cudaMemPrefetchAsync` moves the data in bulk before the launch, so the kernel never waits on page faults.

**Prefetching is better when**:

- The host initializes the data once and the GPU then reads it, as in AXPY, where `X` and `Y` are filled on the CPU and consumed by the kernel.
- The kernel is launched many times on the same data, or the data is large (2^26 and above), where fault-driven migration is expensive.
- You need predictable kernel timing, for example when benchmarking or for real-time work.
- The access pattern is known ahead of time and the data fits in GPU memory.

**Letting CUDA manage memory is better when**:

- The access pattern is irregular or data-dependent. Prefetching everything would move pages the kernel never uses, while on-demand migration moves only the touched pages.
- The data is larger than GPU memory (oversubscription). Prefetching it all can thrash, while page faulting moves pages as needed.
- It is a one-shot or small workload. For small arrays such as n = 1003, prefetch call overhead is comparable to the whole run, and there is little migration to save.
- Developer time matters more than the last bit of speed, because unified memory needs no tuning.
- CPU and GPU alternate on the same data and the next consumer is not known in advance.

### 5. SIMD vs SIMT Performance
**Question:** Between SIMD and SIMT, which one is faster? Give some use cases in which one is better than the other.

   In this AXPY project, SIMT (CUDA) was much faster than SIMD. The best SIMD result was YMM at around 7.36 ms, which is only about 3.23x faster than the C version. The CUDA versions ran at around 0.77 to 0.84 ms, which is about 28x to 31x faster than C. That makes the GPU roughly 9 to 10 times faster than our best SIMD version. It is because AXPY is simple math done on 16.7 million elements, and each element is independent of the others. A GPU has thousands of threads that can each handle a few elements at the same time, while SIMD on the CPU can only process 4 floats (XMM) or 8 floats (YMM) per instruction.

   SIMT is better when the data is very large and every element can be computed on its own, such as vector math, image processing, and matrix operations. It also works best when the data is already on the GPU or can be prefetched, so memory transfer does not slow it down.

   SIMD is better when the data is small, because the GPU has overhead (like the first-run delay that is seen in the unified memory version) that may not be worth it for a small array. It is also better when the program needs low latency or has a lot of branching and sequential logic, since SIMD stays on the CPU and there is no data transfer to the graphics card. Lastly, it is the practical choice when a GPU is not available or the rest of the program already runs on the CPU. For big, simple, data-parallel work like AXPY, SIMT is the better choice. For small data or logic-heavy work, SIMD on the CPU is still a good option.

---

## V. Experience and Methodology
   Comparing the traditional CPU approach with the four GPU implementations shows a dramatic speed improvement when moving to graphics card processing. The standard CPU implementation takes roughly around 23.65 ms to process 16.77 million data elements. By contrast, even a basic GPU configuration drops that time down to around 0.84 ms, which gives us around a 28x speedup. This massive difference happens because GPUs contain thousands of processing cores working simultaneously, making them exceptionally efficient at running simple math calculations across massive batches of data.

   However, getting peak performance out of a GPU requires carefully managing how memory moves between the computer's system RAM and the graphics card. In the basic GPU setup, the graphics card constantly pauses execution to request data on-demand, causing a huge initial delay of over 100 ms on the very first run. By prefetching the data, telling the system to transfer all necessary information to the graphics card before the calculations begin, this startup lag is completely eliminated, bringing execution time down.
	
   It is also worth mentioning the group’s realization, upon testing the outputs, that the outputs might be varying due to our personal machines' speed, even when running the same code. Eventually, we decided to just keep the outputs the same as the ones that made them. This is worth taking into account, as it would probably be a reason for having ‘weird’ comparisons in the outputs. This was also evident in the previous SIMD project, and it helped us realize this factor.

---

## VI. SIMD vs SIMT Handling Parallelism

   While CPU vector processing (SIMD) and GPU parallel execution (SIMT) both process multiple data values at once, they handle the execution differently. SIMD relies on a single CPU core issuing one vector instruction that acts on a fixed block of packed data registers. On the other hand, SIMT uses thousands of lightweight threads running the same program across independent lanes, where the GPU hardware automatically manages instruction issuing, thread grouping, and branch execution. In practice, SIMD is ideal for low-latency sequential logic with mild vectorization, while SIMT excels at massive data-parallel workloads like vector math, though performance drops significantly if memory transfers aren't properly managed.
   
SIMD (Single Instruction, Multiple Data)
PROS:
* Lower latency for individual operations
* No need to move data back and forth over a PCIe bus since memory stays on system RAM.
* Predictable execution without the overhead of managing thousands of concurrent threads.
CONS:
* Limited core count and smaller vector widths yield much lower total throughput for massive datasets.
* Constrained by CPU memory bandwidth limits when processing millions of elements.
  
SIMT (Single Instruction, Multiple Threads)
PROS:
* Massive thread parallelism that provides high speedups on memory-heavy vector math.
* Hardware automatically hides latency by switching to another thread group when one group is waiting on memory.
* Easily scales across large datasets using simple grid loops without needing manual vector register instructions.
CONS:
* Suffers high performance penalties if memory isn't prefetched or managed properly.
* Complex branching or conditional logic inside threads can slow down overall execution speed.








---

