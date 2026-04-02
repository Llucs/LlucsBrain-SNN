/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

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
            #pragma omp atomic
            neurons[s.dst_idx].v_m += s.weight;
        }
    }
}

// Persistência de Dados Fragmentada (Cérebro Fragmentado)
void save_brain_fragmented(const std::string& base_filename, const std::vector<Neuron>& neurons, const std::vector<Synapse>& synapses, size_t max_file_size_mb) {
    const size_t max_bytes = max_file_size_mb * 1024 * 1024;
    uint32_t num_neurons = static_cast<uint32_t>(neurons.size());
    uint32_t num_synapses = static_cast<uint32_t>(synapses.size());

    // 1. Salvar Metadados e Neurônios (Geralmente cabem em um fragmento)
    std::string meta_file = base_filename + "_meta.dat";
    std::ofstream out_meta(meta_file, std::ios::binary);
    out_meta.write(reinterpret_cast<const char*>(&num_neurons), sizeof(uint32_t));
    out_meta.write(reinterpret_cast<const char*>(&num_synapses), sizeof(uint32_t));
    out_meta.write(reinterpret_cast<const char*>(neurons.data()), num_neurons * sizeof(Neuron));
    out_meta.close();

    // 2. Salvar Sinapses Fragmentadas
    size_t synapse_size = sizeof(Synapse);
    size_t synapses_per_fragment = max_bytes / synapse_size;
    uint32_t total_fragments = (num_synapses + synapses_per_fragment - 1) / synapses_per_fragment;

    for (uint32_t f = 0; f < total_fragments; ++f) {
        std::string fragment_name = base_filename + "_part_" + std::to_string(f) + ".dat";
        std::ofstream out_frag(fragment_name, std::ios::binary);
        
        uint32_t start_idx = f * synapses_per_fragment;
        uint32_t end_idx = std::min(start_idx + (uint32_t)synapses_per_fragment, num_synapses);
        uint32_t count = end_idx - start_idx;

        out_frag.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
        out_frag.write(reinterpret_cast<const char*>(&synapses[start_idx]), count * synapse_size);
        out_frag.close();
        
        std::cout << "Fragmento salvo: " << fragment_name << " (" << count << " sinapses)" << std::endl;
    }
}

bool load_brain_fragmented(const std::string& base_filename, std::vector<Neuron>& neurons, std::vector<Synapse>& synapses) {
    std::string meta_file = base_filename + "_meta.dat";
    if (!fs::exists(meta_file)) return false;

    // 1. Carregar Metadados e Neurônios
    std::ifstream in_meta(meta_file, std::ios::binary);
    uint32_t num_neurons, num_synapses;
    in_meta.read(reinterpret_cast<char*>(&num_neurons), sizeof(uint32_t));
    in_meta.read(reinterpret_cast<char*>(&num_synapses), sizeof(uint32_t));
    
    neurons.resize(num_neurons);
    in_meta.read(reinterpret_cast<char*>(neurons.data()), num_neurons * sizeof(Neuron));
    in_meta.close();

    // 2. Carregar Sinapses de todos os fragmentos encontrados
    synapses.clear();
    synapses.reserve(num_synapses);

    uint32_t f = 0;
    while (true) {
        std::string fragment_name = base_filename + "_part_" + std::to_string(f) + ".dat";
        if (!fs::exists(fragment_name)) break;

        std::ifstream in_frag(fragment_name, std::ios::binary);
        uint32_t count;
        in_frag.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        
        size_t current_size = synapses.size();
        synapses.resize(current_size + count);
        in_frag.read(reinterpret_cast<char*>(&synapses[current_size]), count * sizeof(Synapse));
        in_frag.close();
        
        std::cout << "Fragmento carregado: " << fragment_name << std::endl;
        f++;
    }

    if (synapses.size() != num_synapses) {
        std::cerr << "Erro: Número de sinapses carregadas (" << synapses.size() << ") difere do esperado (" << num_synapses << ")" << std::endl;
        return false;
    }

    std::cout << "Cérebro reconstruído com sucesso a partir de " << f << " fragmentos." << std::endl;
    return true;
}
