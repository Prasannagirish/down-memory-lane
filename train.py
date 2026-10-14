"""Train a Flux LoRA on Replicate from ./training_photos.

Usage:
  python train.py            # dry run: builds the zip, prints the plan, spends nothing
  python train.py --launch   # creates the destination model and starts the (paid) training
"""
import os, sys, shutil, json
from dotenv import load_dotenv
import replicate

load_dotenv()
OWNER = replicate.Client().accounts.current().username
DEST_NAME = "down-memory-lane-lora"
TRIGGER = "girish"  # natural trigger (the user's name); the bot rewrites "me/my/myself" to this
TRAINER = "ostris/flux-dev-lora-trainer"
STEPS = 1200

shutil.make_archive("training_dataset", "zip", "dataset")
n = len([f for f in os.listdir("dataset") if f.endswith(".jpg")])
size_mb = os.path.getsize("training_dataset.zip") / 1e6
print(f"zip: {n} images, {size_mb:.1f} MB | destination: {OWNER}/{DEST_NAME} (private) | trigger: {TRIGGER} | steps: {STEPS}")

if "--launch" not in sys.argv:
    print("dry run only; pass --launch to start training")
    sys.exit(0)

try:
    replicate.models.get(f"{OWNER}/{DEST_NAME}")
except replicate.exceptions.ReplicateError:
    replicate.models.create(owner=OWNER, name=DEST_NAME, visibility="private",
                            hardware="gpu-t4", description="Personal Flux LoRA for the Down Memory Lane Slack bot")

version = replicate.models.get(TRAINER).latest_version
with open("training_dataset.zip", "rb") as f:
    training = replicate.trainings.create(
        model=TRAINER, version=version.id, destination=f"{OWNER}/{DEST_NAME}",
        input={"input_images": f, "trigger_word": TRIGGER, "steps": STEPS,
               "autocaption": False,  # captions come from dataset/*.txt (see build_dataset.py)
               "lora_rank": 16, "learning_rate": 0.0004, "resolution": "512,768,1024"},
    )
json.dump({"id": training.id, "destination": f"{OWNER}/{DEST_NAME}", "trigger": TRIGGER}, open("training_run.json", "w"))
print("started training", training.id, "-> https://replicate.com/p/" + training.id)
