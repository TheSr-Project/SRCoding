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
    
    Parameters:
    - site_name: The name of the website/platform to open (e.g. 'youtube', 'google', 'wikipedia', 'github').
    - search_query: Optional search term to look for on that site (e.g. 'cat videos', 'weather today').
    """
    site_name = site_name.lower().strip()
    search_query = search_query.strip()
    
    # URL encode the search query
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
        # Fallback: if it's an unrecognized site name, perform a Google search or open the domain
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
    Examples of common app_names: 'spotify', 'chrome', 'notepad', 'calculator', 'explorer'.
    """
    app_name = app_name.lower().strip()
    
    # Common app mappings to Windows execution methods
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
            # Fallback: try starting it directly via shell command
            # Escape spaces and clean the input
            safe_name = app_name.replace('"', '').replace("'", "")
            subprocess.Popen(f"start {safe_name}", shell=True)
            return f"Attempted to open {app_name} using Windows 'start' command."
    except Exception as e:
        return f"Failed to open {app_name}. Error: {str(e)}"


def close_app(app_name: str) -> str:
    """
    Closes a local running application on the Windows computer.
    Use this tool when the user asks to close, exit, or terminate a program/application.
    Examples of common app_names: 'spotify', 'chrome', 'notepad', 'calculator'.
    """
    app_name = app_name.lower().strip()
    
    # Map friendly names to Windows process names (.exe)
    process_mappings = {
        "spotify": "Spotify.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "browser": "chrome.exe",
        "notepad": "notepad.exe",
        "calculator": "CalculatorApp.exe", # Windows Modern Calc
        "calc": "CalculatorApp.exe",
        "explorer": "explorer.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe"
    }
    
    process_name = process_mappings.get(app_name, f"{app_name}.exe")
    
    try:
        # Run Windows native taskkill command
        cmd = f"taskkill /IM {process_name} /F"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return f"Successfully closed {app_name} ({process_name})."
        else:
            # If default .exe failed, try taskkill directly on the raw name
            cmd_fallback = f"taskkill /IM {app_name} /F"
            result_fallback = subprocess.run(cmd_fallback, shell=True, capture_output=True, text=True)
            if result_fallback.returncode == 0:
                return f"Successfully closed {app_name}."
            return f"Could not find or close {app_name}. Windows message: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed to close {app_name}. Error: {str(e)}"