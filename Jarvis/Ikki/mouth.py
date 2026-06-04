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
        
        # Queues for concurrent pipeline
        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Start worker threads
        self.generator_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.player_thread = threading.Thread(target=self._playback_worker, daemon=True)
        
        self.generator_thread.start()
        self.player_thread.start()

    def prewarm(self):
        self.agent.speak(".", ref_text=self.ref_text, ref_audio_path=self.ref_audio)

    def _generation_worker(self):
        """
        Background worker that reads sentences from sentence_queue,
        generates the audio chunks via self.agent.speak,
        and pushes the compiled audio arrays to audio_queue.
        """
        while True:
            sentence = self.sentence_queue.get()
            if sentence is None:
                # Sentinel to indicate end of current reply session
                self.audio_queue.put(None)
                self.sentence_queue.task_done()
                continue
                
            if not sentence.strip():
                self.sentence_queue.task_done()
                continue
                
            try:
                # Generate audio chunks (this blocks while generating)
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
                print(f"\n[Mouth Error] TTS generation failed: {e}")
            
            self.sentence_queue.task_done()

    def _playback_worker(self):
        """
        Background worker that plays audio chunks from audio_queue sequentially.
        """
        while True:
            audio = self.audio_queue.get()
            if audio is None:
                # End of current reply session
                self.audio_queue.task_done()
                continue
                
            try:
                # Play audio and wait for it to finish playing before starting the next chunk
                sd.play(audio, self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"\n[Mouth Error] Playback failed: {e}")
                
            self.audio_queue.task_done()

    def speak(self, text: str):
        """
        For backward compatibility, plays a full string by pushing it.
        """
        self.speak_sentence(text)
        self.end_session()
        self.wait_until_done()

    def speak_sentence(self, sentence: str):
        """
        Pushes a single sentence into the generation pipeline.
        This call is non-blocking and returns immediately.
        """
        if sentence.strip():
            self.sentence_queue.put(sentence)

    def end_session(self):
        """
        Pushes a sentinel to indicate the end of the current streaming reply.
        """
        self.sentence_queue.put(None)

    def wait_until_done(self):
        """
        Blocks until both queues are fully processed (all sentences generated and spoken).
        """
        self.sentence_queue.join()
        self.audio_queue.join()