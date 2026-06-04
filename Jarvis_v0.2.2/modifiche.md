# Changelog delle Modifiche - Ottimizzazioni Ikki (Jarvis)

Questo documento mostra il confronto tra il codice originale ("Prima") e il codice finale ottimizzato ("Dopo") per ciascuno dei file modificati nel progetto `Ikki`.

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

## 2. brain.py (Streaming Frasale, Rimozione Terminale e Integrazione Windows)

### Prima
```python
from fury import Agent, HistoryManager
from fury.types import create_tool
from tools import get_current_time, open_website, execute_terminal_command

# [Dichiarazioni dei tool precedenti: get_time_tool, open_website_tool, terminal_tool]

class Brain:
    def __init__(self):
        self.agent = Agent(
            model="llama3.1",
            base_url="http://127.0.0.1:11434/v1",
            system_prompt=(
                "You are Ikki, a personal AI assistant. You are witty and conversational. "
                "Always respond in English, in a concise and natural way, as if speaking out loud. "
                "CRITICAL INSTRUCTION: When answering normally (like 'How are you?'), you MUST use PLAIN TEXT. "
                "Do NOT ever output JSON format or tool templates for normal conversation. "
                "ONLY trigger tools if the user explicitly commands you to 'open up' an app, check the time, or search."
                                "ACTION RULE: When the user commands you to do something (like opening/closing apps or searching), you MUST trigger the tool IMMEDIATELY. Do NOT announce 'I am doing it' or 'I will execute this'. Never substitute a real tool execution with conversational text. Just do it."
            ),
            tools=[get_time_tool, open_website_tool, terminal_tool]
        )
        self.history_manager = HistoryManager(agent=self.agent, auto_compact=False)

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
from tools import get_current_time, open_website, open_app, close_app

get_time_tool= create_tool(
    id= "get_current_time",
    description= "ONLY use this tool if the user explicitly asks 'what time is it' or 'tell me the time'. Do NOT use this tool for general greetings like 'good morning' or 'hello'.",
    execute= get_current_time,
    input_schema={
        "type":"object",
        "properties":{},
        "required":[]
    },
    output_schema={}
)

open_website_tool= create_tool (
    id= "open_website",
    description= (
        "ONLY use this tool if the user explicitly asks to open a website, search on Google, search on YouTube, or find a video/song/topic online. "
        "Can perform searches inside the website (e.g. 'open youtube and search for cat videos', or 'search google for python tutorials') using the optional search_query parameter."
    ),
    execute= open_website,
    input_schema={
        "type":"object",
        "properties":{
            "site_name":{"type": "string", "description":"The name of the website or platform to open (e.g. 'youtube', 'google', 'wikipedia', 'github')." },
            "search_query":{"type": "string", "description":"Optional search query to run on that site (e.g. the video title or the search term)." }
        },
        "required":["site_name"]
    },
    output_schema={}
)

open_app_tool = create_tool(
    id="open_app",
    description="ONLY use this tool if the user explicitly asks to 'open', 'run', 'start' or 'launch' a program/application on the computer. Do NOT use this for normal conversation.",
    execute=open_app,
    input_schema={
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "The name of the local application to open (e.g. spotify, chrome, notepad, calculator)."}
        },
        "required": ["app_name"]
    },
    output_schema={}
)

close_app_tool = create_tool(
    id="close_app",
    description="ONLY use this tool if the user explicitly asks to 'close', 'exit', 'terminate' or 'kill' a program/application. Do NOT use this for normal conversation.",
    execute=close_app,
    input_schema={
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "The name of the local application to close (e.g. spotify, chrome, notepad, calculator)."}
        },
        "required": ["app_name"]
    },
    output_schema={}
)


class Brain:
    def __init__(self):
        self.agent = Agent(
            model="llama3.1",
            base_url="http://127.0.0.1:11434/v1",
            system_prompt=(
                "You are Ikki, a personal AI assistant. You are witty and conversational. "
                "Always respond in English, in a concise and natural way, as if speaking out loud. "
                "CRITICAL INSTRUCTION: When answering normally (like 'How are you?'), you MUST use PLAIN TEXT. "
                "Do NOT ever output JSON format or tool templates for normal conversation. "
                "ONLY trigger tools if the user explicitly commands you to 'open' an app, 'close' an app, check the time, or search/open a website."
                "ACTION RULE: When the user commands you to do something (like opening/closing apps or searching), you MUST trigger the tool IMMEDIATELY. Do NOT announce 'I am doing it' or 'I will execute this'. Never substitute a real tool execution with conversational text. Just do it."
                "ENVIRONMENT NOTE: You are running on a Windows system. You do NOT need to write terminal commands yourself. Simply use the high-level tools 'open_app', 'close_app', and 'open_website'."
            ),
            tools=[get_time_tool, open_website_tool, open_app_tool, close_app_tool]
        )
        self.history_manager = HistoryManager(agent=self.agent, auto_compact=False)

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

# [Codice import e variabili d'ambiente]

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

## 4. tools.py (Eliminazione terminale e strumenti specifici Windows)

### Prima
```python
import datetime
import webbrowser
import os
import subprocess

def get_current_time()-> str:
    now= datetime.datetime.now()
    return now.strftime("%H:%M")

def open_website(site_name: str) -> None:
    site_name = site_name.lower().strip()
    urls = {
        "youtube" :  "https://youtube.com",
        "google": "https://google.com",
        "github": "https://github.com"
    }
    url = urls.get(site_name, f"https://google.com/search?q={site_name}")
    webbrowser.open(url)
    return f"I have successfully opened {site_name} on your computer."
    
def execute_terminal_command(command:str) -> str:
    try:
        results =  subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return f"Command executed successfully: {results}"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output}"
```

### Dopo
```python
import datetime
import webbrowser
import os
import subprocess
import urllib.parse

def get_current_time()-> str:
    """
    Returns the current local time. 
    Use this when the user asks what time it is.
    """
    now = datetime.datetime.now()
    return now.strftime("%H:%M")


def open_website(site_name: str, search_query: str = "") -> str:
    """
    Opens a website in the default browser, optionally performing a search query.
    Use this when the user asks to open a site (e.g. YouTube, Google, Wikipedia, Github) or search for something on these platforms.
    """
    site_name = site_name.lower().strip()
    search_query = search_query.strip()
    encoded_query = urllib.parse.quote(search_query)
    
    if site_name == "youtube":
        if search_query:
            url = f"https://www.youtube.com/results?search_query={encoded_query}"
        else:
            url = "https://youtube.com"
    elif site_name == "google":
        if search_query:
            url = f"https://www.google.com/search?q={encoded_query}"
        else:
            url = "https://google.com"
    elif site_name == "wikipedia":
        if search_query:
            url = f"https://it.wikipedia.org/wiki/Speciale:Ricerca?search={encoded_query}"
        else:
            url = "https://it.wikipedia.org"
    elif site_name == "github":
        if search_query:
            url = f"https://github.com/search?q={encoded_query}"
        else:
            url = "https://github.com"
    else:
        if "." in site_name or site_name.startswith("http"):
            url = site_name if site_name.startswith("http") else f"https://{site_name}"
        elif search_query:
            url = f"https://www.google.com/search?q={site_name}+{encoded_query}"
        else:
            url = f"https://www.google.com/search?q={encoded_query}" if search_query else f"https://www.google.com/search?q={site_name}"

    webbrowser.open(url)
    if search_query:
        return f"Successfully opened {site_name} and searched for '{search_query}'."
    return f"Successfully opened {site_name}."


def open_app(app_name: str) -> str:
    """
    Opens a local application on the Windows computer.
    Use this tool when the user asks to open, run, or start a program/application.
    """
    app_name = app_name.lower().strip()
    app_mappings = {
        "spotify": ("protocol", "spotify:"),
        "chrome": ("cmd", "chrome.exe"),
        "google chrome": ("cmd", "chrome.exe"),
        "browser": ("cmd", "chrome.exe"),
        "notepad": ("cmd", "notepad.exe"),
        "calculator": ("cmd", "calc.exe"),
        "calc": ("cmd", "calc.exe"),
        "explorer": ("cmd", "explorer.exe"),
        "files": ("cmd", "explorer.exe"),
        "file explorer": ("cmd", "explorer.exe"),
        "paint": ("cmd", "mspaint.exe"),
        "mspaint": ("cmd", "mspaint.exe")
    }
    
    try:
        if app_name in app_mappings:
            launch_type, value = app_mappings[app_name]
            if launch_type == "protocol":
                os.startfile(value)
            else:
                subprocess.Popen(value, shell=True)
            return f"Successfully opened {app_name}."
        else:
            safe_name = app_name.replace('"', '').replace("'", "")
            subprocess.Popen(f"start {safe_name}", shell=True)
            return f"Attempted to open {app_name} using Windows 'start' command."
    except Exception as e:
        return f"Failed to open {app_name}. Error: {str(e)}"


def close_app(app_name: str) -> str:
    """
    Closes a local running application on the Windows computer.
    """
    app_name = app_name.lower().strip()
    process_mappings = {
        "spotify": "Spotify.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "browser": "chrome.exe",
        "notepad": "notepad.exe",
        "calculator": "CalculatorApp.exe",
        "calc": "CalculatorApp.exe",
        "explorer": "explorer.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe"
    }
    
    process_name = process_mappings.get(app_name, f"{app_name}.exe")
    
    try:
        cmd = f"taskkill /IM {process_name} /F"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully closed {app_name} ({process_name})."
        else:
            cmd_fallback = f"taskkill /IM {app_name} /F"
            result_fallback = subprocess.run(cmd_fallback, shell=True, capture_output=True, text=True)
            if result_fallback.returncode == 0:
                return f"Successfully closed {app_name}."
            return f"Could not find or close {app_name}. Windows message: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed to close {app_name}. Error: {str(e)}"
```

---

## 5. assistant.py (Integrazione della Pipeline Streaming e Ciclo Continuo)

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
import msvcrt
from brain import Brain
from ears import Ears
from mouth import Mouth

async def main():
    print("Inizializzazione sistema in corso...")
    
    brain = Brain()
    ears = Ears()
    mouth = Mouth(agent=brain.agent, sample_rate=24000)
    
    mouth.prewarm()
    
    print("Ikki online. (Parla liberamente! Premi 'q' sulla tastiera in qualsiasi momento per uscire)")
    print("[Premi INVIO una prima volta per iniziare la conversazione...]")
    input()
          
    while True:
        # Verifica se l'utente ha premuto 'q' sulla tastiera prima di avviare la registrazione
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b'q':
                print("\n[Uscita richiesta da tastiera]")
                break
                
        # Avvio registrazione con silenzio dinamico (VAD)
        print("\n[Ascolto... parla ora]")
        audio_b64 = ears.listen()
        
        # Verifica tastiera subito dopo la registrazione
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b'q':
                print("\n[Uscita richiesta da tastiera]")
                break

        transcript = await brain.transcribe_audio(audio_b64)
        if not transcript:
            # Se nessun audio viene rilevato, riascolta senza rispondere
            continue
         
        print(f"Utente: {transcript}")
        
        # Verifica per comandi vocali di uscita
        exit_commands = {"exit", "quit", "goodbye", "shut down", "fermati", "esci", "addio"}
        if any(cmd in transcript.lower() for cmd in exit_commands):
            print("Ikki: Goodbye!")
            mouth.speak("Goodbye!")
            break
            
        print("Ikki: ", end="", flush=True)
        
        # Invio frasi nello stream a bocca dinamica man mano che arrivano
        async for sentence in brain.think_and_respond():
            mouth.speak_sentence(sentence)
            
        mouth.end_session()
        
        # Attesa asincrona del completamento del parlato
        await asyncio.to_thread(mouth.wait_until_done)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Programma terminato con Ctrl+C]")
```
