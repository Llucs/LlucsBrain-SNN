/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <cmath>
#include <fstream>
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <device_launch_parameters.h>

__global__ void update_neurons_kernel(Neuron* neurons, float* external_current, SimulationParams params, float current_time, uint32_t num_neurons) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neurons) return;

    Neuron& n = neurons[idx];
    n.fired = false;

    if (n.refractory_timer > 0.0f) {
        n.refractory_timer -= params.dt;
        n.v_m = params.v_reset;
        return;
    }

    float dv = (-(n.v_m - params.v_rest) + params.r_m * external_current[idx]) / params.tau_m;
    n.v_m += dv * params.dt;

    if (n.v_m >= params.v_th) {
        n.v_m = params.v_reset;
        n.last_spike_time = current_time;
        n.refractory_timer = params.t_ref;
        n.spike_count++;
        n.fired = true;
    }
}

__global__ void process_spikes_kernel(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    Synapse& s = synapses[idx];
    Neuron& pre = neurons[s.src_idx];
    Neuron& post = neurons[s.dst_idx];

    if (pre.fired) {
        atomicAdd(&(post.v_m), s.weight);
    }
}
#endif

// Funções de interface para CPU/GPU
void update_neurons(Neuron* neurons, float* external_current, SimulationParams params, float current_time, uint32_t num_neurons) {
#ifdef USE_CUDA
    int threadsPerBlock = 256;
    int blocks = (num_neurons + threadsPerBlock - 1) / threadsPerBlock;
    update_neurons_kernel<<<blocks, threadsPerBlock>>>(neurons, external_current, params, current_time, num_neurons);
#else
    #pragma omp parallel for if(USE_OPENMP)
    for (uint32_t i = 0; i < num_neurons; ++i) {
        Neuron& n = neurons[i];
        n.fired = false;
        if (n.refractory_timer > 0.0f) {
            n.refractory_timer -= params.dt;
            n.v_m = params.v_reset;
        } else {
            float dv = (-(n.v_m - params.v_rest) + params.r_m * external_current[i]) / params.tau_m;
            n.v_m += dv * params.dt;
            if (n.v_m >= params.v_th) {
                n.v_m = params.v_reset;
                n.last_spike_time = current_time;
                n.refractory_timer = params.t_ref;
                n.spike_count++;
                n.fired = true;
            }
        }
    }
#endif
}

void process_spikes(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params) {
#ifdef USE_CUDA
    int threadsPerBlock = 256;
    int blocks = (num_synapses + threadsPerBlock - 1) / threadsPerBlock;
    process_spikes_kernel<<<blocks, threadsPerBlock>>>(neurons, synapses, num_synapses, params);
#else
    #pragma omp parallel for if(USE_OPENMP)
    for (uint32_t i = 0; i < num_synapses; ++i) {
        Synapse& s = synapses[i];
        if (neurons[s.src_idx].fired) {
            // No CPU, podemos usar atomic ou garantir que não haja race conditions
            // Para simplicidade e performance em CPU pura, usamos atomic se disponível ou loop serial
            #pragma omp atomic
            neurons[s.dst_idx].v_m += s.weight;
        }
    }
#endif
}

// Persistência de Dados
void save_checkpoint(const std::string& filename, const std::vector<Neuron>& neurons, const std::vector<Synapse>& synapses) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Erro ao salvar checkpoint: " << filename << std::endl;
        return;
    }
    uint32_t num_neurons = neurons.size();
    uint32_t num_synapses = synapses.size();
    out.write(reinterpret_cast<const char*>(&num_neurons), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&num_synapses), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(neurons.data()), num_neurons * sizeof(Neuron));
    out.write(reinterpret_cast<const char*>(synapses.data()), num_synapses * sizeof(Synapse));
    out.close();
    std::cout << "Checkpoint salvo: " << filename << std::endl;
}

bool load_checkpoint(const std::string& filename, std::vector<Neuron>& neurons, std::vector<Synapse>& synapses) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        return false;
    }
    uint32_t num_neurons, num_synapses;
    in.read(reinterpret_cast<char*>(&num_neurons), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_synapses), sizeof(uint32_t));
    neurons.resize(num_neurons);
    synapses.resize(num_synapses);
    in.read(reinterpret_cast<char*>(neurons.data()), num_neurons * sizeof(Neuron));
    in.read(reinterpret_cast<char*>(synapses.data()), num_synapses * sizeof(Synapse));
    in.close();
    std::cout << "Checkpoint carregado: " << filename << " (" << num_neurons << " neurônios)" << std::endl;
    return true;
}
