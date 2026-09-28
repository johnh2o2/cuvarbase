// Reuse the installed native CUB agent and its floating-point association.
// Runtime guards and a startup canary live in tls_reference_short_prefix.py.
#include <cub/version.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cuda/std/functional>

#if CUB_VERSION != 200800
#error "This native scan wrapper requires CUB 2.8.0"
#endif
#if __CUDACC_VER_MAJOR__ != 12 || __CUDACC_VER_MINOR__ != 4 || __CUDACC_VER_BUILD__ != 131
#error "Only nvcc 12.4.131 has been validated for this native scan wrapper"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 860
#error "Only SM86 policy parity has been source-audited"
#endif

using Op = cuda::std::plus<>;
using Policy = cub::detail::scan::policy_hub<float, Op>::Policy860::ScanPolicyT;
using Agent = cub::detail::scan::AgentScan<
    Policy, float*, float*, Op, cub::NullType, unsigned int, float, false>;
static_assert(Policy::BLOCK_THREADS == 128, "native thread count changed");
static_assert(Policy::ITEMS_PER_THREAD == 15, "native tile mapping changed");
static_assert(Policy::SCAN_ALGORITHM == cub::BLOCK_SCAN_WARP_SCANS,
              "native scan association changed");
static_assert(Policy::LOAD_ALGORITHM == cub::BLOCK_LOAD_WARP_TRANSPOSE,
              "native load policy changed");
static_assert(Policy::STORE_ALGORITHM == cub::BLOCK_STORE_WARP_TRANSPOSE,
              "native store policy changed");

extern "C" __global__ __launch_bounds__(128)
void native_cub_short_rows(float* input, float* output, int rows, int columns)
{
    const int row = blockIdx.x;
    if (row >= rows || columns < 1 || columns > Agent::TILE_ITEMS) return;
    __shared__ typename Agent::TempStorage temporary;
    typename Agent::ScanTileStateT unused_state;
    // This is exactly the dynamic device scan's first AND last tile branch.
    // No lookback/state access occurs for tile_idx=0, IS_LAST_TILE=true.
    // CUB also preserves its native first-element fill of unused suffix slots.
    const long long offset = (long long)row * columns;
    Agent agent(temporary, input + offset, output + offset, Op{}, cub::NullType{});
    agent.template ConsumeTile<true>((unsigned int)columns, 0, 0, unused_state);
}
