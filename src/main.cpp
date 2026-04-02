/**
 * Desenvolvido por Llucs
 */

#include "LlucsBrain.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <omp.h>

int main(int argc, char** argv) {
    const uint32_t num_neurons = 1000000;
    const uint32_t num_synapses = 10000000;
    const int num_steps = 1000;
    const std::string checkpoint_file = "brain_state.dat";

    SimulationParams params;
    params.dt = 1.0f;
    params.v_rest = -70.0f;
    params.v_th = -50.0f;
    params.v_reset = -75.0f;
    params.tau_m = 20.0f;
    params.r_m = 1.0f;
    params.t_ref = 2.0f;
    params.leak_decay = 0.95f;
    params.stdp_a_plus = 0.01f;
    params.stdp_a_minus = 0.012f;
    params.stdp_tau_plus = 20.0f;
    params.stdp_tau_minus = 20.0f;
    params.w_max = 1.0f;
    params.w_min = 0.0f;

    std::cout << "LlucsBrain Engine - Evolução Automatizada (CPU OpenMP)" << std::endl;
    std::cout << "Threads OpenMP disponíveis: " << omp_get_max_threads() << std::endl;

    std::vector<Neuron> h_neurons;
    std::vector<Synapse> h_synapses;
    std::vector<float> h_external_current(num_neurons);

    // Tentar carregar checkpoint existente
    if (!load_checkpoint(checkpoint_file, h_neurons, h_synapses)) {
        std::cout << "Nenhum checkpoint encontrado. Inicializando nova rede..." << std::endl;
        h_neurons.resize(num_neurons);
        h_synapses.resize(num_synapses);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist_v(-75.0f, -50.0f);
        std::uniform_int_distribution<uint32_t> dist_idx(0, num_neurons - 1);
        std::uniform_real_distribution<float> dist_w(0.1f, 0.5f);

        for (uint32_t i = 0; i < num_neurons; ++i) {
            h_neurons[i].v_m = dist_v(gen);
            h_neurons[i].last_spike_time = -100.0f;
            h_neurons[i].refractory_timer = 0.0f;
            h_neurons[i].spike_count = 0;
            h_neurons[i].fired = false;
        }

        for (uint32_t i = 0; i < num_synapses; ++i) {
            h_synapses[i].src_idx = dist_idx(gen);
            h_synapses[i].dst_idx = dist_idx(gen);
            h_synapses[i].weight = dist_w(gen);
            h_synapses[i].last_pre_spike = -100.0f;
            h_synapses[i].last_post_spike = -100.0f;
        }
    }

    // Inicializar corrente externa aleatória para cada execução
    std::mt19937 gen_curr(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist_curr(0.0f, 15.0f);
    for (uint32_t i = 0; i < num_neurons; ++i) {
        h_external_current[i] = dist_curr(gen_curr);
    }

    std::cout << "Executando simulação paralela em CPU..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; ++step) {
        float current_time = step * params.dt;
        
        // 1. Atualizar Neurônios (LIF)
        update_neurons(h_neurons.data(), h_external_current.data(), params, current_time, num_neurons);
        
        // 2. Processar Spikes (Propagação Sináptica)
        process_spikes(h_neurons.data(), h_synapses.data(), num_synapses, params);
        
        // 3. Aplicar Aprendizado (STDP)
        apply_stdp(h_neurons.data(), h_synapses.data(), num_synapses, params, current_time);

        if (step % 100 == 0) {
            std::cout << "Passo " << step << "/" << num_steps << " concluído." << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulação finalizada em " << elapsed.count() << " segundos." << std::endl;

    // Salvar estado final
    save_checkpoint(checkpoint_file, h_neurons, h_synapses);

    return 0;
}
