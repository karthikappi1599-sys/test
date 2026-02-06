from datasets import load_dataset, Dataset, Audio
import os
import gc
from huggingface_hub import HfApi
import os

# Set token only for this script (not saved globally)
os.environ["HF_TOKEN"] = "hf_your_write_token_here"

api = HfApi(token=os.environ["HF_TOKEN"])

gc.set_threshold(700, 10, 10)

print("Loading full dataset into memory (this will take time + disk space)...")

ds = load_dataset(
    "simon3000/genshin-voice",
    split="train",
    streaming=False,
    cache_dir=None
)

ds = ds.cast_column("audio", Audio(decode=False))

print(f"Dataset loaded. Number of examples: {len(ds):,}")

speakers = ["Nahida", "Lynette", "Paimon", "Citlali"]

# -------------------------
# FIX: no lambda with num_proc
# -------------------------
def is_japanese(row):
    return row.get("language") == "Japanese" and row.get("speaker") in speakers

def is_english(row):
    return row.get("language") == "English(US)" and row.get("speaker") in speakers

print("Filtering Japanese lines...")
japanese = ds.filter(
    is_japanese,
    num_proc=4,
    desc="Filtering Japanese"
)

print(f"Japanese filtered: {len(japanese):,} examples")

print("Filtering English(US) lines...")
english = ds.filter(
    is_english,
    num_proc=4,
    desc="Filtering English"
)

print(f"English filtered: {len(english):,} examples")

japanese.push_to_hub(
    "Karthikappi0011/genshin-japanese-topfull",
    private=False,
    token=os.environ["HF_TOKEN"]
)

english.push_to_hub(
    "Karthikappi0011/genshin-english-topfull",
    private=False,
    token=os.environ["HF_TOKEN"]
)


print("All done! Check https://huggingface.co/Karthikappi0011")
