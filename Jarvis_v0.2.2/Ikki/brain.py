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