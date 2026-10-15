# Fantasy AI

A language model fine-tuned to think and write in fantasy: new worlds, new
magic systems, new myths. It's meant for drafting novels and movie scripts.

## How it works

Training a model from scratch on fantasy books alone won't produce a good
writer. All the fantasy ever published is only a few billion words, and
models learn grammar, logic and dialogue from far more text than that. The
practical approach:

1. **Start from a fluent open-weights model** (Llama 3.1 8B, Qwen 2.5, Mistral…).
2. **Fine-tune it with QLoRA** on a fantasy corpus so it picks up the genre's
   voice, pacing, imagery and archetypes.
3. **Prompt it with your own world** (names, maps, magic rules) so it invents
   new stories instead of retelling old ones.

```
data/sources.yaml ──► download_gutenberg.py ──► data/raw/public_domain/*.txt ─┐
                                               data/raw/licensed/*.txt ───────┤
                                                                              ▼
                     generate.py ◄── outputs/fantasy-ai-lora ◄── train_lora.py ◄── prepare_dataset.py
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_gutenberg.py     # ~30 public-domain fantasy & myth classics
python scripts/prepare_dataset.py        # clean + chunk -> data/processed/*.jsonl
python scripts/train_lora.py             # needs a CUDA GPU (24 GB for 8B; Colab T4 for 3B)
python scripts/generate.py --prompt "The bone-witches of Harrowmere did not bury their dead."
python scripts/generate.py --mode screenplay --prompt "EXT. SUNKEN CITADEL - NIGHT"
```

Using Llama models requires accepting Meta's license on Hugging Face and
running `huggingface-cli login`. Pass `--base-model Qwen/Qwen2.5-7B` to use a
model without that gate.

No GPU? Rent one by the hour (RunPod, Lambda, Vast.ai) or use Google Colab.
One epoch over this corpus on a single 4090 takes a few hours.

## About the books you listed

*Lord of the Rings, Harry Potter, A Song of Ice and Fire, The Witcher,
Shannara* and the scripts for *Supernatural, The Originals, Into the
Badlands* are all **under copyright**, so:

- **Don't download pirated copies.** In *Bartz v. Anthropic* (2025) the court
  found that training on books that were bought legally could be fair use,
  but training on pirated copies was not. That case ended in a $1.5B
  settlement.
- **Don't commit or share the text.** `.gitignore` already blocks
  `data/raw/**/*.txt`.
- **Don't publish the model if it was trained on them.** A model fine-tuned
  on one author can reproduce long passages, and then you are distributing
  their book.

What you *can* do:

- **Personal, private experiments:** put ebooks you bought and converted to
  `.txt` in `data/raw/licensed/`. The pipeline picks them up automatically.
  How far this is allowed depends on your country and the ebook's terms, so
  check before you rely on it, and keep the resulting model to yourself.
- **Learn from the same roots they did.** Tolkien drew on *Beowulf*, the
  *Eddas*, the *Kalevala* and William Morris. Sapkowski drew on Slavic folk
  tales and the Grimms. Supernatural draws on Dracula, Carmilla and folklore.
  All of these are public domain and already listed in `data/sources.yaml`.
- **Add your own writing**, and CC-licensed fantasy (some authors release
  books under Creative Commons). Your own worldbuilding notes are the best
  data you can add.

You want new fantasies, not remixed Middle-earth, so a model that learned
the genre's deep patterns instead of one franchise's names will do this job
better anyway.

## Roadmap

- [x] Public-domain corpus downloader
- [x] Dataset cleaning & chunking
- [x] QLoRA fine-tuning script
- [x] Generation script (novel / screenplay modes)
- [ ] Screenplay-format data (self-written or licensed) for proper script output
- [ ] Instruction stage: "Write chapter 3 where…" style prompt/response pairs
- [ ] "World bible" support: feed a world's lore, characters and rules into every prompt
- [ ] Long-form planner: outline → chapters → scenes, so it can sustain a whole novel
- [ ] Simple web UI
