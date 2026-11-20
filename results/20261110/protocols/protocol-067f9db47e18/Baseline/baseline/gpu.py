"""Track actual allocated GPU service sessions, not per-request latency."""
from datetime import datetime, timezone
from .io import read_jsonl, append_jsonl


def gpu_event(path, action, session, *, gpu_count=1, cap=40):
    events = read_jsonl(path)
    starts = {e["session"]: e for e in events if e["action"] == "start"}
    stops = {e["session"]: e for e in events if e["action"] == "stop"}
    now = datetime.now(timezone.utc)
    hours = sum(((datetime.fromisoformat(stops[key]["utc"]) if key in stops else now) - datetime.fromisoformat(e["utc"])).total_seconds() / 3600 * e["gpu_count"] for key, e in starts.items())
    if action == "status":
        return {"gpu_hours": hours, "cap_hours": cap, "open_sessions": sorted(set(starts) - set(stops)), "remaining_hours": max(0, cap - hours)}
    if action == "start":
        if hours >= cap or gpu_count != 1 or session in starts or set(starts) - set(stops):
            raise ValueError("40-hour/single-GPU cap, duplicate session, or another open service allocation")
    elif action == "stop":
        if session not in starts or session in stops:
            raise ValueError("No matching open GPU session")
    else:
        raise ValueError("Unsupported GPU ledger action")
    append_jsonl(path, {"action": action, "session": session, "utc": now.isoformat(), "gpu_count": gpu_count})
    return gpu_event(path, "status", session, cap=cap)
