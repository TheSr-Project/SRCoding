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