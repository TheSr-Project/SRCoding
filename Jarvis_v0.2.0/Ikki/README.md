# Progetto Ikki 🎙️🤖

Ikki è un assistente vocale locale e intelligente sviluppato per Windows 11, basato sul pacchetto Fury ed ottimizzato per sfruttare l'accelerazione hardware NVIDIA CUDA (RTX 5070 12GB).

## Architettura del Progetto

Il progetto è suddiviso in moduli indipendenti pensati per facilitare l'apprendimento e il debugging:

1. **`brain.py` (Il Cervello)**: Gestisce la logica LLM locale tramite Ollama e Fury-SDK.
2. **`ears.py` (L'Udito)**: Gestisce la registrazione dal microfono e lo Speech-to-Text tramite `faster-whisper`.
3. **`mouth.py` (La Voce)**: Gestisce il Text-to-Speech con clonazione vocale zero-shot tramite NeuTTS.
4. **`assistant.py` (L'Orchestratore)**: Unisce i moduli in un ciclo continuo di conversazione vocale.
5. **`skills/` (Le Abilità)**: Strumenti personalizzati per consentire a Ikki di controllare il sistema operativo Windows.
