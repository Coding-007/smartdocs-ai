import  warnings
warnings.filterwarnings("ignore")

# Simple in-memory store
_sessions = {}

def get_history(session_id: str) -> list:
    return _sessions.get(session_id, [])

def add_to_history(session_id: str, role: str, content: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []

    # Add new message to history
    _sessions[session_id].append({
        "role": role,
        "content": content
    })

    # Keep only last 10 messages per session
    if len(_sessions[session_id]) > 10:
        _sessions[session_id] = _sessions[session_id][-10:]

def clear_history(session_id: str) -> None:
    if session_id in _sessions:
        del _sessions[session_id]

def get_all_sessions() -> list:
    return list(_sessions.keys())