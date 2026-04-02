/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>

// Funções de interface para CPU com OpenMP
void update_neurons(Neuron* neurons, float* external_current, SimulationParams params, float current_time, uint32_t num_neurons) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < num_neurons; ++i) {
        Neuron& n = neurons[i];
        n.fired = false;

        if (n.refractory_timer > 0.0f) {
            n.refractory_timer -= params.dt;
            n.v_m = params.v_reset;
        } else {
            // Equação diferencial LIF: dv/dt = (-(v - v_rest) + R*I) / tau_m
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
}

void process_spikes(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < num_synapses; ++i) {
        Synapse& s = synapses[i];
        if (neurons[s.src_idx].fired) {
            // Uso de atomic para evitar race conditions no potencial de membrana do neurônio pós-sináptico
            #pragma omp atomic
            neurons[s.dst_idx].v_m += s.weight;
        }
    }
}

// Persistência de Dados
void save_checkpoint(const std::string& filename, const std::vector<Neuron>& neurons, const std::vector<Synapse>& synapses) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Erro ao salvar checkpoint: " << filename << std::endl;
        return;
    }
    uint32_t num_neurons = static_cast<uint32_t>(neurons.size());
    uint32_t num_synapses = static_cast<uint32_t>(synapses.size());
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
