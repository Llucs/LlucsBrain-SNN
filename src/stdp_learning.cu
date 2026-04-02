/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <device_launch_parameters.h>

__global__ void apply_stdp_kernel(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params, float current_time) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    Synapse& s = synapses[idx];
    Neuron& pre = neurons[s.src_idx];
    Neuron& post = neurons[s.dst_idx];

    if (post.fired) {
        float dt = current_time - s.last_pre_spike;
        if (dt > 0.0f) {
            float dw = params.stdp_a_plus * expf(-dt / params.stdp_tau_plus);
            s.weight = fminf(params.w_max, s.weight + dw);
        }
        s.last_post_spike = current_time;
    }

    if (pre.fired) {
        float dt = current_time - s.last_post_spike;
        if (dt > 0.0f) {
            float dw = params.stdp_a_minus * expf(-dt / params.stdp_tau_minus);
            s.weight = fmaxf(params.w_min, s.weight - dw);
        }
        s.last_pre_spike = current_time;
    }
}
#endif

void apply_stdp(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params, float current_time) {
#ifdef USE_CUDA
    int threadsPerBlock = 256;
    int blocks = (num_synapses + threadsPerBlock - 1) / threadsPerBlock;
    apply_stdp_kernel<<<blocks, threadsPerBlock>>>(neurons, synapses, num_synapses, params, current_time);
#else
    #pragma omp parallel for if(USE_OPENMP)
    for (uint32_t i = 0; i < num_synapses; ++i) {
        Synapse& s = synapses[i];
        Neuron& pre = neurons[s.src_idx];
        Neuron& post = neurons[s.dst_idx];

        if (post.fired) {
            float dt = current_time - s.last_pre_spike;
            if (dt > 0.0f) {
                float dw = params.stdp_a_plus * std::exp(-dt / params.stdp_tau_plus);
                s.weight = std::min(params.w_max, s.weight + dw);
            }
            s.last_post_spike = current_time;
        }

        if (pre.fired) {
            float dt = current_time - s.last_post_spike;
            if (dt > 0.0f) {
                float dw = params.stdp_a_minus * std::exp(-dt / params.stdp_tau_minus);
                s.weight = std::max(params.w_min, s.weight - dw);
            }
            s.last_pre_spike = current_time;
        }
    }
#endif
}
