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

---

## I. Program Output and Correctness Check
*(Screenshot of the program output with correctness check AND execution time for all cases)*

[Insert Screenshot(s) Here]

## II. NSYS Profiling
*(Screenshot of nsys for all CUDA variants)*

**Variant 2: No Prefetch, No Mem Advise**
[Insert Screenshot Here]

**Variant 3: With Prefetch, No Page Creation, No Mem Advise**
[Insert Screenshot Here]

**Variant 4: With Prefetch, With Page Creation, No Mem Advise**
[Insert Screenshot Here]

**Variant 5: With Prefetch, With Page Creation, With Mem Advise**
[Insert Screenshot Here]

---

## III. Comparative Table of Execution Time

| 2^24 elements | | |
| :--- | :--- | :--- |
| **CUDA block size = [256 \| 512 \| 1024]** | **Kernel time [avg \| min \| max] in ms** | **Speedup vs baseline C program** |
| C | [23.780529 \| 19.297711 \| 28.231964] | N/A |
| x86-64 | [ \| min \| max] | [value] |
| x86-64 SIMD XMM | [avg \| min \| max] | [value] |
| x86-64 SIMD YMM | [avg \| min \| max] | [value] |
| CUDA Unified (No Prefetch/Advise) | [0.836048 \| 0.831090 \| 101.798247] | [28.44] |
| CUDA Prefetch (No Page Creation) | [avg \| min \| max] | [value] |
| CUDA Prefetch + Page Creation | 256: [0.770405 \| 0.768544 \| 0.772672]<br>512: [0.775305 \| 0.772448 \| 0.784384]<br>1024: [0.782987 \| 0.780864 \| 0.791168] | 256: [30.87]<br>512: [30.67]<br>1024: [30.37] |
| CUDA Prefetch + Page Creation + Mem Advise| [avg \| min \| max] | [value] |

*(Note: Add more tables if needed for different vector sizes or block sizes)*

---

## IV. Analysis of Results

### 1. Analysis of Kernel Execution Time
[Discuss the execution times observed across the different CUDA variants and baseline programs. Identify trends, minimums, maximums, and averages.]

### 2. Analysis of Speed Performance Across All Platforms
[Compare the speedup across C, x86-64, SIMD (XMM/YMM), and the CUDA variants. Which platform performed the best and why?]

### 3. Impact of Block Size
**Question:** How does block size affect execution time (observing various elements and using max blocks)? Which block size will you recommend?

[Provide your analysis and recommendation here...]

### 4. Memory Management: Prefetching vs. CUDA Managed
**Question:** Is prefetching always recommended, or should CUDA manage memory? Give some use cases in which one is better than the other.

[Provide your analysis and use cases here...]

### 5. SIMD vs SIMT Performance
**Question:** Between SIMD and SIMT, which one is faster? Give some use cases in which one is better than the other.

[Provide your comparison and use cases here...]

---

## V. Experience and Methodology
*(Discuss the problems encountered and solutions made, unique methodology used, AHA moments, etc.)*

[Discuss your team's experience, roadblocks, debugging processes, and any specific insights gained during development.]

---

## VI. SIMD vs SIMT Handling Parallelism
*(Discuss, based on your experience on the particular project use case, the differences between SIMD and SIMT in handling parallelism. Include also the PROS and CONS of using SIMD and SIMT in your use case.)*

**Differences:**
[Discuss architectural and practical differences observed...]

**SIMD (Single Instruction, Multiple Data)**
* **PROS:** [List pros...]
* **CONS:** [List cons...]

**SIMT (Single Instruction, Multiple Threads)**
* **PROS:** [List pros...]
* **CONS:** [List cons...]







---

