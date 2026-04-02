/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <cmath>
#include <algorithm>
#include <omp.h>

void apply_stdp(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params, float current_time) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < num_synapses; ++i) {
        Synapse& s = synapses[i];
        Neuron& pre = neurons[s.src_idx];
        Neuron& post = neurons[s.dst_idx];

        // LTP (Long-Term Potentiation): Reforço baseado no disparo pós-sináptico
        if (post.fired) {
            float dt = current_time - s.last_pre_spike;
            if (dt > 0.0f) {
                float dw = params.stdp_a_plus * std::exp(-dt / params.stdp_tau_plus);
                s.weight = std::min(params.w_max, s.weight + dw);
            }
            s.last_post_spike = current_time;
        }

        // LTD (Long-Term Depression): Enfraquecimento baseado no disparo pré-sináptico
        if (pre.fired) {
            float dt = current_time - s.last_post_spike;
            if (dt > 0.0f) {
                float dw = params.stdp_a_minus * std::exp(-dt / params.stdp_tau_minus);
                s.weight = std::max(params.w_min, s.weight - dw);
            }
            s.last_pre_spike = current_time;
        }
    }
}
