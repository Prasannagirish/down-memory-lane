"""Down Memory Lane: a Slack bot that turns "me at 5 on a beach" into a photo of you as a child.

Flow: Slack message -> parse age/scene (prompting.py) -> Flux LoRA on Replicate (imagegen.py) -> image back in the same thread.
Runs over Socket Mode, so it needs no public URL.
"""
import os, time, logging, threading
from collections import defaultdict, deque
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()
from prompting import build_request, RequestRejected
from imagegen import generate_image, log_generation, ModelNotReady

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("down-memory-lane")
app = App(token=os.environ["SLACK_BOT_TOKEN"])

RATE_LIMIT, RATE_WINDOW = int(os.getenv("RATE_LIMIT", "8")), 3600     # cost control: images per user per hour
_recent: dict[str, deque] = defaultdict(deque)
_seen_events: set[str] = set()                                        # Slack can redeliver events; never bill twice
_lock = threading.Lock()


def _allow(user: str) -> bool:
    now = time.time()
    with _lock:
        q = _recent[user]
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return False
        q.append(now)
        return True


def handle_request(client, channel: str, user: str, text: str, thread_ts: str, event_key: str) -> None:
    with _lock:
        if event_key in _seen_events:
            return
        _seen_events.add(event_key)

    def say(msg, **kw):
        return client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=msg, **kw)

    try:
        req = build_request(text)
    except RequestRejected as e:
        say(str(e)); log_generation(user=user, ok=False, reason="rejected", text=text); return
    if not _allow(user):
        say(f"You've hit the limit of {RATE_LIMIT} photos per hour. Come back a little later! :hourglass_flowing_sand:")
        log_generation(user=user, ok=False, reason="rate_limited"); return

    age_note = f" (I picked age {req.age}; say e.g. _me at 5_ to choose)" if req.age_was_guessed and req.age < 18 else ""
    ack = say(f":camera_with_flash: Rolling back the years to age {req.age}… this takes about 20 seconds.{age_note}")
    try:
        image, secs = generate_image(req)
    except ModelNotReady:
        client.chat_update(channel=channel, ts=ack["ts"], text="My photo album is still being set up. Try again in a few minutes!")
        log_generation(user=user, ok=False, reason="model_not_ready"); return
    except Exception:
        log.exception("generation failed")
        client.chat_update(channel=channel, ts=ack["ts"], text=":warning: Something went wrong developing that photo. Please try again.")
        log_generation(user=user, ok=False, reason="error", text=text); return

    client.files_upload_v2(channel=channel, thread_ts=thread_ts, file=image, filename=f"down-memory-lane-age{req.age}.jpg",
                           title=f"Age {req.age}: {req.scene}", initial_comment=f":sparkles: Here you are at {req.age}, {req.scene}!")
    client.chat_delete(channel=channel, ts=ack["ts"])
    log_generation(user=user, ok=True, age=req.age, scene=req.scene, seconds=round(secs, 1), prompt=req.prompt)
    log.info("delivered age=%s in %.1fs", req.age, secs)


@app.event("app_mention")
def on_mention(event, client):
    handle_request(client, event["channel"], event["user"], event["text"], event.get("thread_ts") or event["ts"], event["ts"])


@app.event("message")
def on_message(event, client):
    if event.get("channel_type") != "im" or event.get("subtype") or event.get("bot_id"):
        return
    handle_request(client, event["channel"], event["user"], event["text"], event.get("thread_ts") or event["ts"], event["ts"])


if __name__ == "__main__":
    log.info("Down Memory Lane is listening…")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
