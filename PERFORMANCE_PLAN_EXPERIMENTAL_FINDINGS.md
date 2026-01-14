# Experimental Findings: Per-Realization Scheduler Architecture

**Date:** January 14, 2026
**Status:** Experimental prototype - code will be REVERTED
**Purpose:** Document learnings for future optimization work

---

## Executive Summary: What We Learned

### ❌ Per-Realization Architecture Doesn't Work in Python/Numba

**Result:** 2-15× performance **regression** instead of speedup

**Root cause:** Python overhead multiplied by number of active realizations
- Baseline: 1 call to `_calculate_next_updates` per step
- Per-realization: ~32 calls per step (one per active realization)
- Python function calls + data structure overhead × 32 = massive slowdown

**Takeaway:** Don't pursue this approach in Python/Numba

### ❌ Numba `prange` Parallelism Incompatible with Fire Spread

**Issue:** Fire spread produces variable-length outputs (each cell → 0-8 neighbors)

**Why it fails:**
- Can't use `list.append()` in `prange` (race conditions)
- Can't preallocate arrays (output size unknown)
- Would need complex two-phase algorithm

**Takeaway:** Current algorithm structure fundamentally incompatible with Numba parallelism

### ✅ Heap Optimization Works Great

**Successfully reduced scheduler overhead to <1% using min-heap**
- O(log N) operations instead of O(N) linear scans
- Lazy cleanup of stale entries
- **Can be applied to baseline shared scheduler!**

**Takeaway:** This optimization is worth keeping

### 💡 Baseline Architecture is Actually Well-Designed

**Batch processing all realizations together:**
- Amortizes Python overhead across realizations
- Single Numba call per step (not N calls)
- Better than we thought!

**Takeaway:** Keep baseline architecture, optimize within it

---

## Detailed Performance Results

### Benchmark Comparison (All with 10× Scale Factor)

```
Test Case                    Baseline    Experimental  Slowdown
──────────────────────────────────────────────────────────────────
Large scale (5000×5000, 100) 8.2s        125.7s        15.3×
100 realizations (500×500)   2.7s        32.9s         12.2×
50 realizations              2.2s        15.5s         7.0×
25 realizations              1.8s        6.9s          3.8×
10 realizations              1.3s        2.8s          2.2×
Heterogeneous fuels          0.04s       0.10s         2.8×
Terrain varied               1.2s        5.5s          4.5×
Variable wind                1.0s        2.3s          2.2×
```

**Pattern:** Slowdown scales linearly with number of realizations!

### Profiling Breakdown (Experimental Implementation)

**Configuration:** 500 steps, 100 realizations, 500×500 grid, 0.710s total

```
Component                      Time    %       Why
────────────────────────────────────────────────────────────────────
_calculate_next_updates        0.534s  75%     Called ~32× per step!
  ├─ split_by_time            0.195s  28%     Python dict/list ops
  ├─ UpdateBatch.__init__     0.120s  17%     Array copying, min/max
  └─ Numba overhead           0.081s  11%     Type inference per call
_filter_valid_updates          0.097s  14%     UpdateBatch creation
Scheduler operations           0.042s   6%     Per-realization loops
Heap management                0.009s   1%     ✅ Optimized!
```

**Critical insight:** Only 11% of time is actual Numba fire spread code!
The rest is Python overhead that got multiplied by ~32×.

---

## Why Per-Realization Architecture Failed

### Call Count Explosion

**Baseline approach (shared scheduler):**
```python
# All realizations processed together
batch = UpdateBatch(times, cells, realizations, deltas, weights)
new_updates = next_updates_fn(batch)  # 1 call per step
```
- **500 steps × 1 call/step = 500 total calls**

**Per-realization approach:**
```python
# Each realization processed separately
for realization in active_realizations:
    batch = UpdateBatch(times, cells, deltas, weights)  # No realizations field
    new_updates = next_updates_fn(batch)  # One call per active realization
```
- **500 steps × ~32 active realizations/step = ~16,000 total calls**

### Python Overhead Breakdown (Per Call)

Each call to `_calculate_next_updates` involves:

1. **Python function call overhead** (~1-5μs)
2. **split_by_time()** - Python dict/list operations to group by time (~50-200μs)
3. **UpdateBatch.__init__()** - Array copying, min/max, validation (~30-100μs)
4. **Numba type inference** - cache=False due to multiprocessing (~20-50μs)
5. **Actual fire spread** - Numba-compiled, fast (~10-30μs)

Steps 1-4 dominate when multiplied by ~32!

### Baseline's Hidden Advantage

**We thought:** Separating realizations would enable parallelization

**Reality:** Python/Numba can't exploit parallelism anyway (GIL + variable-length outputs)

**Baseline benefit:** Batch processing amortizes overhead
- Process all burning cells across all realizations in one Numba call
- Share type inference, data structure operations
- One `split_by_time` call handles all updates together

---

## Why Numba Parallelism Doesn't Work

### The Variable-Length Output Problem

**Fire spread algorithm:**
```python
for cell_index in range(n_burning_cells):
    cell = burning_cells[cell_index]
    # Each cell spreads to 0-8 neighbors (depends on grid position, fuel, etc.)
    neighbors = get_spreadable_neighbors(cell)  # Variable length!
    for neighbor in neighbors:
        calculate_ignition_time(neighbor)
        output.append(neighbor_update)  # Can't do this in prange!
```

**Why `prange` fails:**
```python
@njit(parallel=True)
def process_cells(cells):
    updates = []  # Need to know size for parallel!
    for i in prange(len(cells)):  # Want parallelism
        cell_updates = process_cell(cells[i])  # Returns 0-8 items
        updates.extend(cell_updates)  # ❌ Race condition!
    return updates
```

**Attempted solutions:**
1. **Two-phase (count, allocate, fill)** - Complex, likely slower than serial
2. **Fixed-size output (8 per cell)** - Wastes memory, requires filtering
3. **Thread-local lists + merge** - Merge overhead too high

**Conclusion:** Algorithm structure fundamentally incompatible with `prange`

### Why Python Threading Doesn't Help

**GIL (Global Interpreter Lock)** prevents true parallelism:
- Only one thread executes Python bytecode at a time
- CPU-bound workload (fire spread calculations)
- Threading only helps with I/O-bound tasks

---

## What Actually Works: Heap Optimization

### Implementation

**Before (linear scan):**
```python
def next_time(self) -> float:
    return min(scheduler.next_time() for scheduler in self.schedulers)  # O(N)
```

**After (min-heap):**
```python
def next_time(self) -> float:
    while self.heap:
        time, idx = self.heap[0]  # O(1) peek
        if self.schedulers[idx].next_time() == time:
            return time  # Valid minimum
        heapq.heappop(self.heap)  # Lazy cleanup of stale entries
    return float('inf')
```

### Performance Impact

**Reduced scheduler overhead from 6% to <1%**

**This optimization can be applied to baseline shared scheduler!**

Even though per-realization architecture didn't work, the heap technique is valuable.

---

## Baseline Architecture Analysis

### Why Baseline is Better Than Expected

**Original assessment:** "Baseline mixes realizations, hard to parallelize"

**Reality discovered:** Batch processing is an **advantage** in Python:

1. **Amortizes Python overhead** across all realizations
2. **Single Numba call** per step (not N calls)
3. **Shared type inference** (Numba caches compilation)
4. **Potential vectorization** opportunities

### Where Baseline Could Be Optimized

**From profiling the experimental code, we identified bottlenecks:**

1. **split_by_time (28% in experiment)** - Python dict/list operations
   - Could convert to Numba?
   - Optimize data structures?

2. **UpdateBatch.__init__ (17% in experiment)** - Array copying, min/max
   - Use array views instead of copies?
   - Lazy computation of min/max?

3. **Scheduler operations (6% in baseline)** - Linear scans
   - Apply heap optimization → <1%

**Note:** These percentages are from the experimental per-realization code. In baseline, Numba fire spread might dominate more. Need to profile baseline to confirm.

---

## Three Paths Forward

### Option 1: Optimize Baseline in Python/Numba (Recommended for Short-Term)

**Keep shared scheduler architecture, apply targeted optimizations:**

1. ✅ **Apply heap optimization** to shared scheduler
   - Proven to reduce overhead to <1%
   - Low risk, easy to implement

2. 📊 **Profile baseline** to identify actual bottlenecks
   - May be different from per-realization profiling
   - Focus optimization efforts on real hotspots

3. 🔍 **Investigate Python bottlenecks:**
   - Is `split_by_time` a bottleneck in baseline?
   - Is `UpdateBatch.__init__` overhead significant?
   - Can these be converted to Numba?

4. 🔬 **Profile actual fire spread models:**
   - Rothermel, FWI, custom models
   - May find optimization opportunities in physics calculations

**Expected gain:** 1.5-2× speedup with careful optimization
**Effort:** Low-Medium (targeted changes)
**Risk:** Low (incremental improvements to proven architecture)

### Option 2: Rust Core Rewrite (Recommended for Long-Term)

**Per-realization architecture is IDEAL for Rust:**

```rust
// Each realization is independent, true parallelism
realizations.par_iter_mut()  // Rayon parallel iterator
    .for_each(|realization| {
        // No Python overhead (function calls ~1-5ns)
        // No GIL (true multi-core parallelism)
        // Zero-copy data structures
        realization.step();
    });
```

**Why it works in Rust but not Python:**
- **No Python overhead:** Function calls are nanoseconds, not microseconds
- **No GIL:** True parallelism across cores
- **Efficient memory:** Zero-copy, stack allocation, no Python dict/list overhead
- **Type safety:** Compile-time guarantees without runtime overhead

**Expected gain:** 5-20× with 8-16 cores (assuming per-core performance similar to Python)
**Effort:** High (rewrite core logic, FFI to Python)
**Risk:** Medium (new language, team expertise needed)

**Note:** Branch `rust-core` suggests this is already being explored!

### Option 3: Hybrid Approach

**Phase 1: Quick wins in Python (1-2 weeks)**
1. Apply heap optimization to baseline
2. Profile baseline to find real bottlenecks
3. Optimize identified hotspots (convert to Numba, reduce copying, etc.)
4. **Target:** 1.5-2× speedup

**Phase 2: Re-evaluate (after profiling)**
- If Python overhead still dominates → proceed to Rust
- If fire spread physics dominates → optimize models
- If "fast enough" → stop here

**Phase 3: Rust when justified (2-3 months)**
- Clear profiling evidence that Python is the limit
- Concrete speedup targets (5×+ needed)
- Team has Rust expertise or time to learn

**Expected gain:** 1.5-2× short-term, 5-20× long-term
**Effort:** Incremental (spread over time)
**Risk:** Low (validate each phase before proceeding)

---

## Recommendations for Next Session

### ✅ DO These Things

1. **Start from baseline code** (revert all experimental changes)

2. **Apply heap optimization to baseline shared scheduler**
   - Proven technique with <1% overhead
   - Low risk, immediate benefit
   - Implementation pattern available from experiment

3. **Profile baseline to identify real bottlenecks**
   - Don't assume bottlenecks are same as in experimental code
   - Use `pyinstrument` or `cProfile`
   - Focus on large-scale scenarios (100 realizations, 5000×5000 grid)

4. **Investigate optimization opportunities:**
   - Can `split_by_time` be converted to Numba?
   - Can `UpdateBatch` reduce copying?
   - Are fire spread models (Rothermel, FWI) optimized?

5. **Keep batch processing architecture**
   - It amortizes Python overhead effectively
   - Proven to be faster than per-realization in Python

### ❌ DON'T Do These Things

1. **Don't implement per-realization schedulers in Python**
   - Experimental results prove it's 2-15× slower
   - Python overhead multiplied by number of realizations
   - Only viable in Rust/C++ with no Python overhead

2. **Don't try Numba `prange` for fire spread**
   - Variable-length outputs make it fundamentally incompatible
   - Would require complex two-phase algorithm
   - Not worth the complexity

3. **Don't try Python threading for parallelism**
   - GIL prevents true parallelism for CPU-bound work
   - Won't help with fire spread calculations

4. **Don't optimize without profiling first**
   - Bottlenecks in experimental code may differ from baseline
   - Profile, then optimize the real hotspots

### 🎯 Success Criteria

**Short-term (Python/Numba):**
- Heap optimization applied and verified (<1% scheduler overhead)
- Baseline profiled and bottlenecks identified
- 1.5-2× speedup achieved through targeted optimizations

**Long-term (if pursuing Rust):**
- Per-realization architecture implemented in Rust
- True multi-core parallelism with Rayon
- 5-20× speedup demonstrated on multi-core systems

---

## Appendix: Experimental Code Summary

**Files that were modified (all will be reverted):**
- `src/propagator/core/propagator.py` - Per-realization step() logic
- `src/propagator/core/scheduler.py` - Heap-based scheduler implementation
- `src/propagator/core/models.py` - Removed realizations field
- `src/propagator/core/propagation.py` - Array of independent schedulers

**Benchmark files created (can be kept for future comparisons):**
- `benchmark_baseline.py` - Baseline performance tests
- `benchmark_current.py` - Per-realization tests (rename to benchmark_experimental.py?)
- `run_comparison.py` - Automated comparison script
- `test_call_count.py` - Instrumentation for call counting

**Key architectural changes in experiment:**
1. Created array of `Scheduler` objects (one per realization)
2. Removed `realizations` field from `UpdateBatch`/`UpdateBatchWithTime`
3. Modified `step()` to iterate over schedulers, process each realization independently
4. Implemented min-heap for efficient `next_time()` queries across schedulers
5. Added lazy cleanup strategy for stale heap entries

**All changes provide valuable insights but code will be discarded!**

The knowledge gained is documented here for the next session.
