[README.md](https://github.com/user-attachments/files/26448579/README.md)
# LlucsBrain Engine - Evolução Automatizada

Desenvolvido por Llucs

Este é um motor de simulação de cérebro biológico de alto desempenho, implementado em C++ e CUDA, com suporte a execução híbrida (CPU/GPU) e persistência de dados para evolução contínua via GitHub Actions.

## Características Técnicas

- **Modelo de Neurônio:** Leaky Integrate-and-Fire (LIF) com física de membrana realista.
- **Aprendizado:** Spike-Timing-Dependent Plasticity (STDP) para plasticidade sináptica.
- **Arquitetura Híbrida:** 
  - **GPU:** Utiliza CUDA C++ para processamento paralelo massivo.
  - **CPU:** Fallback automático para C++ multithreaded (OpenMP) em ambientes sem GPU.
- **Persistência:** Sistema de checkpoints (`brain_state.dat`) para salvar e carregar o estado sináptico.
- **Automação:** Workflow do GitHub Actions configurado para evolução diária do cérebro.

## Estrutura do Projeto

- `include/LlucsBrain.cuh`: Definições de estruturas e interfaces.
- `src/core_engine.cu`: Kernels LIF e lógica de propagação de spikes (Híbrido).
- `src/stdp_learning.cu`: Lógica de aprendizado STDP (Híbrido).
- `src/main.cpp`: Ponto de entrada, gerenciamento de memória e persistência.
- `.github/workflows/brain_evolution.yml`: Automação para GitHub.

## Instruções de Compilação

Para compilar o projeto:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

O CMake detectará automaticamente a presença do CUDA Toolkit. Se não for encontrado, o motor será compilado em modo CPU (OpenMP).

## Execução e Evolução

Para rodar localmente:
```bash
./LlucsBrain
```

O programa criará ou atualizará o arquivo `brain_state.dat` ao final de cada execução. No GitHub, o workflow `brain_evolution.yml` executará a simulação diariamente e salvará o progresso automaticamente no repositório.

## Créditos

**Desenvolvido por Llucs**
