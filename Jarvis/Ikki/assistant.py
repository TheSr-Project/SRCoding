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
        
        # Start recording with dynamic silence detection
        print("[Ascolto... parla ora]")
        audio_b64 = ears.listen()
        
        transcript = await brain.transcribe_audio(audio_b64)
        if not transcript:
            print("[Nessun audio o trascrizione rilevata]")
            continue
         
        print(f"Utente: {transcript}")
        print("Ikki: ", end="", flush=True)
        
        # Stream sentences from brain and speak them dynamically as they arrive
        async for sentence in brain.think_and_respond():
            mouth.speak_sentence(sentence)
            
        mouth.end_session()
        
        # Wait asynchronously for the speech playback to complete
        await asyncio.to_thread(mouth.wait_until_done)

if __name__ == "__main__":
    asyncio.run(main())