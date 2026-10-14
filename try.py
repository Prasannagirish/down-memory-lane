"""Test the full text -> image path without Slack:  python try.py "me at 5 on a beach" [out.jpg]"""
import sys
from dotenv import load_dotenv
load_dotenv()
from prompting import build_request
from imagegen import generate_image
req = build_request(sys.argv[1])
print("prompt:", req.prompt)
img, secs = generate_image(req, seed=int(sys.argv[3]) if len(sys.argv) > 3 else None)
out = sys.argv[2] if len(sys.argv) > 2 else "try_out.jpg"
open(out, "wb").write(img); print(f"saved {out} ({secs:.1f}s)")
