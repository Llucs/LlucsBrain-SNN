/**
 * Desenvolvido por Llucs
 */

#ifndef LLUCS_BRAIN_CUH
#define LLUCS_BRAIN_CUH

#include <cstdint>
#include <string>
#include <vector>

// Parâmetros de Simulação
struct SimulationParams {
    float dt;               // Passo de tempo (ms)
    float v_rest;           // Potencial de repouso (mV)
    float v_th;             // Limiar de disparo (mV)
    float v_reset;          // Potencial de reset (mV)
    float tau_m;            // Constante de tempo da membrana (ms)
    float r_m;              // Resistência da membrana (GOhm)
    float t_ref;            // Tempo de refração (ms)
    float leak_decay;       // Fator de decaimento
    
    // Parâmetros STDP
    float stdp_a_plus;      // Reforço (LTP)
    float stdp_a_minus;     // Enfraquecimento (LTD)
    float stdp_tau_plus;    // Constante de tempo LTP (ms)
    float stdp_tau_minus;   // Constante de tempo LTD (ms)
    float w_max;            // Peso sináptico máximo
    float w_min;            // Peso sináptico mínimo
};

// Estrutura do Neurônio LIF (Leaky Integrate-and-Fire)
struct Neuron {
    float v_m;              // Potencial de membrana atual
    float last_spike_time;  // Tempo do último disparo
    float refractory_timer; // Tempo restante em refração
    uint32_t spike_count;   // Contador total de disparos
    bool fired;             // Flag de disparo no passo atual
};

// Estrutura de Sinapse (Matriz Esparsa em Formato COO)
struct Synapse {
    uint32_t src_idx;       // Índice do neurônio pré-sináptico
    uint32_t dst_idx;       // Índice do neurônio pós-sináptico
    float weight;           // Peso da conexão
    float last_pre_spike;   // Tempo do último spike pré-sináptico (para STDP)
    float last_post_spike;  // Tempo do último spike pós-sináptico (para STDP)
};

// Funções do Motor (Paralelizadas com OpenMP)
void update_neurons(Neuron* neurons, float* external_current, SimulationParams params, float current_time, uint32_t num_neurons);
void process_spikes(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params);
void apply_stdp(Neuron* neurons, Synapse* synapses, uint32_t num_synapses, SimulationParams params, float current_time);

// Persistência de Dados
void save_checkpoint(const std::string& filename, const std::vector<Neuron>& neurons, const std::vector<Synapse>& synapses);
bool load_checkpoint(const std::string& filename, std::vector<Neuron>& neurons, std::vector<Synapse>& synapses);

#endif // LLUCS_BRAIN_CUH
