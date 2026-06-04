from typing import AsyncGenerator
from fury import Agent, HistoryManager
from fury.types import create_tool
from tools import get_current_time, open_website, execute_terminal_command



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
    description= "ONLY use this tool if the user explicitly says 'open [website]' or 'search google for [query]'. Do NOT use this for normal conversation.",
    execute= open_website,
    input_schema={
        "type":"object",
        "properties":{
            "site_name":{"type": "string", "description":"The name of the website to open (e.g. youtube, google)." }},
        "required":["site_name"]
        
    },
    output_schema={}
)

terminal_tool = create_tool(
    id="execute_terminal_command",
    description="Executes a raw command in the Windows terminal. "
        "WINDOWS COMMAND CHEAT SHEET: "
        "- To open an app: start [appname] "
        "- To close an app: taskkill /IM [appname].exe /F "
        "- To check running processes: tasklist "
        "- To open a folder: explorer [path] "
        "CRITICAL: You MUST provide the exact Windows command string in the 'command' parameter.",
    execute=execute_terminal_command,
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The exact Windows command to run (e.g. 'taskkill /IM spotify.exe /F' or 'start calc')."
            }
        },
        "required": ["command"]
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
                "ONLY trigger tools if the user explicitly commands you to 'open up' an app, check the time, or search."
                                "ACTION RULE: When the user commands you to do something (like opening/closing apps or searching), you MUST trigger the tool IMMEDIATELY. Do NOT announce 'I am doing it' or 'I will execute this'. Never substitute a real tool execution with conversational text. Just do it."
            ),
            tools=[get_time_tool, open_website_tool, terminal_tool]
            
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