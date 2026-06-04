# Changelog delle Modifiche - Ottimizzazioni Ikki (Jarvis)

Questo documento mostra il confronto tra il codice originale ("Prima") e il codice ottimizzato ("Dopo") per ciascuno dei file modificati nel progetto `Ikki`.

---

## 1. ears.py (Voice Activity Detection - VAD)

### Prima
```python
import base64
import io
import sounddevice as sd
import soundfile as sf

class Ears:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def listen(self, duration: float = 5.0) -> str:
        frames = int(duration * self.sample_rate)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()

        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

### Dopo
```python
import base64
import io
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

class Ears:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def listen(self, max_duration: float = 8.0, silence_duration: float = 1.2, silence_threshold: float = 0.01) -> str:
        """
        Records from the microphone until silence is detected or max_duration is reached.
        Uses a simple RMS-based Voice Activity Detection (VAD).
        """
        # Block size for the input stream (e.g. 100ms blocks)
        block_size = int(self.sample_rate * 0.1)
        audio_blocks = []
        
        # VAD State Variables
        started_speaking = False
        silence_start_time = None
        start_time = time.time()
        
        def callback(indata, frames, time_info, status):
            nonlocal started_speaking, silence_start_time
            if status:
                pass # ignore buffer underflow/overflow indications for silent running
            
            # Copy the incoming data
            audio_blocks.append(indata.copy())
            
            # Calculate Root-Mean-Square (RMS) energy of this chunk
            rms = np.sqrt(np.mean(indata**2))
            
            if not started_speaking:
                # If energy is above threshold, user started speaking
                if rms > silence_threshold:
                    started_speaking = True
            else:
                # If user has started speaking and energy drops below threshold, track silence
                if rms < silence_threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                else:
                    silence_start_time = None

        # Create input stream
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, blocksize=block_size, dtype="float32"):
            while True:
                time.sleep(0.05)
                # Check maximum duration timeout
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    break
                
                # Check silence duration after speaking started
                if started_speaking and silence_start_time is not None:
                    silence_elapsed = time.time() - silence_start_time
                    if silence_elapsed >= silence_duration:
                        break
                
                # Check initial silence timeout (if user hasn't started speaking at all after 3.0 seconds)
                if not started_speaking and elapsed >= 3.0:
                    break

        if not audio_blocks:
            return ""
            
        audio = np.concatenate(audio_blocks, axis=0)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

---

## 2. brain.py (Async Generator per le frasi)

### Prima
```python
from fury import Agent, HistoryManager
from fury.types import create_tool
from tools import get_current_time, open_website, execute_terminal_command

# [Definizioni dei tool omesse per brevità]

class Brain:
    def __init__(self):
        # [Codice init omesso]
        pass

    async def transcribe_audio(self, audio_b64: str) -> str:
        await self.history_manager.add_voice(audio_b64)
        return self.history_manager.history[-1]["content"].strip()

    async def think_and_respond(self) -> str:
        reply = ""
        runner = self.agent.runner()
        async for event in runner.chat(self.history_manager.history):
            if event.content:
                reply += event.content
                print(event.content, end="", flush=True)
        print()
        await self.history_manager.add({"role": "assistant", "content": reply})
        return reply
```

### Dopo
```python
from typing import AsyncGenerator
from fury import Agent, HistoryManager
from fury.types import create_tool
from tools import get_current_time, open_website, execute_terminal_command

# [Definizioni dei tool omesse per brevità]

class Brain:
    def __init__(self):
        # [Codice init omesso]
        pass

    async def transcribe_audio(self, audio_b64: str) -> str:
        await self.history_manager.add_voice(audio_b64)
        return self.history_manager.history[-1]["content"].strip()

    async def think_and_respond(self) -> AsyncGenerator[str, None]:
        reply = ""
        runner = self.agent.runner()
        buffer = ""
        sentence_endings = {'.', '?', '!'}
        
        async for event in runner.chat(self.history_manager.history):
            if event.content:
                content = event.content
                buffer += content
                reply += content
                print(content, end="", flush=True)
                
                while True:
                    first_idx = -1
                    for ending in sentence_endings:
                        idx = buffer.find(ending)
                        if idx != -1:
                            if first_idx == -1 or idx < first_idx:
                                first_idx = idx
                                
                    newline_idx = buffer.find('\n')
                    if newline_idx != -1 and (first_idx == -1 or newline_idx < first_idx):
                        first_idx = newline_idx
                    
                    if first_idx == -1:
                        break
                        
                    is_sentence_boundary = False
                    if first_idx == len(buffer) - 1:
                        if buffer[first_idx] == '\n':
                            is_sentence_boundary = True
                        else:
                            break
                    else:
                        next_char = buffer[first_idx + 1]
                        if next_char.isspace():
                            is_sentence_boundary = True
                        elif buffer[first_idx] == '\n':
                            is_sentence_boundary = True
                            
                    if is_sentence_boundary:
                        sentence = buffer[:first_idx + 1].strip()
                        buffer = buffer[first_idx + 1:]
                        if sentence:
                            yield sentence
                    else:
                        break
                        
        print()
        remaining = buffer.strip()
        if remaining:
            yield remaining
            
        await self.history_manager.add({"role": "assistant", "content": reply})
```

---

## 3. mouth.py (Parallelismo TTS e Riproduzione Audio)

### Prima
```python
import os
import numpy as np
import sounddevice as sd
from fury import Agent

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

class Mouth:
    def __init__(self, agent: Agent, sample_rate: int = 24000):
        self.agent = agent
        self.sample_rate = sample_rate
        self.ref_audio = "C:/Users/utente/Desktop/Personal/Jarvis/fury-sdk/examples/resources/ref.wav"
        self.ref_text = "Welcome home sir."

    def prewarm(self):
        self.agent.speak(".", ref_text=self.ref_text, ref_audio_path=self.ref_audio)

    def speak(self, text: str):
        if not text.strip():
            return
            
        audio_chunks = list(
            self.agent.speak(
                text=text,
                ref_text=self.ref_text,
                ref_audio_path=self.ref_audio,
            )
        )
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            sd.play(audio, self.sample_rate)
            sd.wait()
```

### Dopo
```python
import os
import queue
import threading
import numpy as np
import sounddevice as sd
from fury import Agent

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

class Mouth:
    def __init__(self, agent: Agent, sample_rate: int = 24000):
        self.agent = agent
        self.sample_rate = sample_rate
        self.ref_audio = "C:/Users/utente/Desktop/Personal/Jarvis/fury-sdk/examples/resources/ref.wav"
        self.ref_text = "Welcome home sir."
        
        # Queues per la pipeline concorrente
        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Avvio dei thread worker in background
        self.generator_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.player_thread = threading.Thread(target=self._playback_worker, daemon=True)
        
        self.generator_thread.start()
        self.player_thread.start()

    def prewarm(self):
        self.agent.speak(".", ref_text=self.ref_text, ref_audio_path=self.ref_audio)

    def _generation_worker(self):
        """
        Thread in background che estrae le frasi dalla coda,
        genera l'audio tramite self.agent.speak e invia l'audio compilato.
        """
        while True:
            sentence = self.sentence_queue.get()
            if sentence is None:
                self.audio_queue.put(None)
                self.sentence_queue.task_done()
                continue
                
            if not sentence.strip():
                self.sentence_queue.task_done()
                continue
                
            try:
                audio_chunks = list(
                    self.agent.speak(
                        text=sentence,
                        ref_text=self.ref_text,
                        ref_audio_path=self.ref_audio,
                    )
                )
                if audio_chunks:
                    audio = np.concatenate(audio_chunks)
                    self.audio_queue.put(audio)
            except Exception as e:
                print(f"\n[Mouth Error] Generazione TTS fallita: {e}")
            
            self.sentence_queue.task_done()

    def _playback_worker(self):
        """
        Thread in background che riproduce sequenzialmente i blocchi audio.
        """
        while True:
            audio = self.audio_queue.get()
            if audio is None:
                self.audio_queue.task_done()
                continue
                
            try:
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"\n[Mouth Error] Riproduzione audio fallita: {e}")
                
            self.audio_queue.task_done()

    def speak(self, text: str):
        """
        Per compatibilità con il vecchio codice, riproduce l'intero testo.
        """
        self.speak_sentence(text)
        self.end_session()
        self.wait_until_done()

    def speak_sentence(self, sentence: str):
        """
        Invia una singola frase nella pipeline. Non bloccante.
        """
        if sentence.strip():
            self.sentence_queue.put(sentence)

    def end_session(self):
        """
        Invia il segnale di fine sessione dello stream corrente.
        """
        self.sentence_queue.put(None)

    def wait_until_done(self):
        """
        Attende che tutte le frasi siano state elaborate e parlate.
        """
        self.sentence_queue.join()
        self.audio_queue.join()
```

---

## 4. assistant.py (Integrazione della Pipeline Streaming)

### Prima
```python
import asyncio
from brain import Brain
from ears import Ears
from mouth import Mouth


async def main():
    
     print("Inizializzazione sistema in corso...")
     
     
     brain =  Brain()
     ears = Ears()
     mouth = Mouth(agent= brain.agent, sample_rate= 24000)
     
     mouth.prewarm()
     
     print("Ikki online. (Premi INVIO per parlare, 'q' per uscire)")
           
     while True: 
          cmd = input("\n>").strip()
          if cmd.lower() in ["q", "quit", "exit"]:
              break
          
          audio_b64= ears.listen(duration=5.0)
          
          transcript =await brain.transcribe_audio(audio_b64)
          if not transcript:
               continue
           
          print(f"Utente: {transcript}")
          print("Ikki: ", end="", flush=True)
          reply = await brain.think_and_respond()
          mouth.speak(reply)
          
          
if __name__ == "__main__":
    asyncio.run(main())
```

### Dopo
```python
import asyncio
from brain import Brain
from ears import Ears
from mouth import Mouth

async def main():
    print("Inizializzazione sistema in corso...")
    
    brain = Brain()
    ears = Ears()
    mouth = Mouth(agent=brain.agent, sample_rate=24000)
    
    mouth.prewarm()
    
    print("Ikki online. (Premi INVIO per parlare, 'q' per uscire)")
          
    while True: 
        cmd = input("\n>").strip()
        if cmd.lower() in ["q", "quit", "exit"]:
            break
        
        # Avvio registrazione con silenzio dinamico (VAD)
        print("[Ascolto... parla ora]")
        audio_b64 = ears.listen()
        
        transcript = await brain.transcribe_audio(audio_b64)
        if not transcript:
            print("[Nessun audio o trascrizione rilevata]")
            continue
         
        print(f"Utente: {transcript}")
        print("Ikki: ", end="", flush=True)
        
        # Invio frasi nello stream a bocca dinamica man mano che arrivano
        async for sentence in brain.think_and_respond():
            mouth.speak_sentence(sentence)
            
        mouth.end_session()
        
        # Attesa asincrona del completamento del parlato
        await asyncio.to_thread(mouth.wait_until_done)

if __name__ == "__main__":
    asyncio.run(main())
```
