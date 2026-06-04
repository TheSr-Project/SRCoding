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
        # Check if user pressed 'q' on keyboard before starting recording
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b'q':
                print("\n[Uscita richiesta da tastiera]")
                break
                
        # Start recording with dynamic silence detection
        print("\n[Ascolto... parla ora]")
        audio_b64 = ears.listen()
        
        # Check keyboard again right after recording
        if msvcrt.kbhit():
            key = msvcrt.getch().lower()
            if key == b'q':
                print("\n[Uscita richiesta da tastiera]")
                break

        transcript = await brain.transcribe_audio(audio_b64)
        if not transcript:
            # If no audio or speech detected, loop again without responding
            continue
         
        print(f"Utente: {transcript}")
        
        # Check for voice exit commands (Italian and English)
        exit_commands = {"exit", "quit", "goodbye", "shut down", "fermati", "esci", "addio"}
        if any(cmd in transcript.lower() for cmd in exit_commands):
            print("Ikki: Goodbye!")
            mouth.speak("Goodbye!")
            break
            
        print("Ikki: ", end="", flush=True)
        
        # Stream sentences from brain and speak them dynamically as they arrive
        async for sentence in brain.think_and_respond():
            mouth.speak_sentence(sentence)
            
        mouth.end_session()
        
        # Wait asynchronously for the speech playback to complete
        await asyncio.to_thread(mouth.wait_until_done)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Programma terminato con Ctrl+C]")