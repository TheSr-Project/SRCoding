# Ikki (Jarvis) - Project Architecture Summary

Questo documento sintetizza l'architettura, le tecnologie e le scelte di design del progetto "Ikki", un assistente vocale locale basato sull'infrastruttura di Jarvis, progettato per operare in totale privacy (in locale) tramite modelli Open Source.

## 1. Stack Tecnologico
* **LLM Core**: `llama3.1` (8B parametri) eseguito in locale tramite **Ollama**. Modello scelto per l'ottimo bilanciamento tra velocità di inferenza e capacità di ragionamento logico/Tool Calling.
* **Orchestrazione Agente**: `fury-sdk`. Libreria utilizzata per gestire la memoria (HistoryManager) e il ciclo di esecuzione degli strumenti (Native Tool Calling).
* **Speech-to-Text (Ears)**: Whisper (modello `base.en`) per la trascrizione in tempo reale dell'audio del microfono.
* **Text-to-Speech (Mouth)**: NeuTTS Nano (modello `neuphonic/neutts-nano-q4-gguf`) eseguito via `onnxruntime` per la generazione vocale a bassa latenza.

## 2. Architettura del Codice (Object-Oriented)
Il monolite iniziale è stato refattorizzato in moduli indipendenti per garantire scalabilità:
* `assistant.py`: L'Orchestratore principale. Inizializza i moduli, gestisce il loop asincrono di ascolto del microfono e fa dialogare i componenti.
* `brain.py`: Contiene la classe `Brain`. Inizializza l'Agente `fury`, definisce il System Prompt, mappa i Tool (strumenti) e gestisce la generazione della risposta e l'esecuzione autonoma delle azioni.
* `ears.py`: Gestisce l'hardware audio in ingresso (`sounddevice`) e codifica la voce in Base64 per la trascrizione.
* `mouth.py`: Espone i metodi per far parlare l'agente sfruttando il motore NeuTTS.
* `tools.py`: Raccoglie le logiche pure Python dei superpoteri dell'agente.

## 3. Tool Calling (L'Agente Reattivo)
A differenza degli agenti basati su grafi rigidi (es. LangGraph), Ikki sfrutta un'architettura **reattiva**. Il modello ha accesso a un set di strumenti (Tool) definiti tramite JSON Schema e decide autonomamente in tempo reale quando invocarli per interagire con l'OS Windows.

### Strumenti implementati:
1. `get_current_time`: Interroga il modulo `datetime`.
2. `open_website`: Sfrutta `webbrowser` per aprire siti specifici o effettuare ricerche su Google.
3. `open_app`: Sfrutta `os.system("start ...")` e protocolli nativi (es. `spotify:`) per lanciare app locali.
4. `close_app`: Esegue `subprocess` con comandi Windows nativi (`taskkill /IM appname.exe /F`) per terminare processi.

*Nota di design strutturale*: Per supportare l'infrastruttura di un modello "leggero" (8B), le complessità della sintassi del terminale Windows (es. i flag di taskkill) sono state relegate al codice Python in `tools.py`. Il LLM deve semplicemente passare il parametro "nome app", abbassando drasticamente il carico cognitivo e azzerando le allucinazioni tecniche.

## 4. Prompt Engineering Avanzato
Per forzare il LLM a comportarsi da Esecutore Silenzioso piuttosto che da Chatbot prolisso, il System Prompt è stato calibrato con regole psicologiche ferree:
* **Silent Execution Rule**: Se l'utente richiede un'azione, il LLM è obbligato a innescare il tool in silenzio senza pre-annunciarlo testualmente. Questo risolve il problema delle "Action Hallucinations" (dire "Lo sto facendo" senza lanciare la funzione).
* **JSON Leakage Prevention**: Istruzioni esplicite vietano al modello di stampare formati JSON nella conversazione testuale di default, costringendolo a separare nettamente le interazioni sociali (Plain Text) dalle esecuzioni tecniche (Tool Calling Schema).

## 5. Prossimi Sviluppi Previsti
* Passaggio dall'interazione a turni a un flusso **Streaming Bidirezionale Real-Time** (modello LiveKit).
* Implementazione della **Memoria a Lungo Termine** (RAG/SQLite) persistente oltre la sessione corrente.
