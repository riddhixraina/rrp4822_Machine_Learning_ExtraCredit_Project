#!/usr/bin/env python
"""
End-to-end scaling study on symbolic music (Lakh MIDI -> text) with 100M tokens per model.

Pipeline:
1) Download Lakh MIDI subset ZIP from URL (or use local MIDI dir).
2) Convert MIDI -> simple text event format using pretty_midi.
3) Clean and load the text, build char tokenizer; compute stats.
4) Build streaming dataset that cycles until 100M tokens per model.
5) Train 5 decoder-only Transformers and 4 LSTMs (1 epoch each on 100M tokens).
6) Track training curves, wall-clock, GPU memory; fit scaling laws and plot.
7) Evaluate best model (test perplexity, 10 samples, ABC validity, MIDI success).

NOTE:
- You MUST set LMD_ZIP_URL to a manageable subset you prepared
  (e.g., 10k–30k MIDI files zipped). The official Lakh full dataset is huge.
"""

import os, math, time, json, random, zipfile, glob
from pathlib import Path

import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pretty_midi
from music21 import converter, midi as m21_midi

# ---------------------- Config ----------------------

class Cfg:
    LMD_ZIP_PATH = "/home/yourname/data/lakh_subset.zip"  # <-- your local zip
    DATA_ROOT = "./data_lakh"
    MIDI_ROOT = "./data_lakh/midi"
    ABC_ROOT  = "./data_lakh/abc"

    USE_LOCAL_MIDI = False           # set True if you already have MIDI extracted
    LOCAL_MIDI_ROOT = "/path/to/local/midi"  # only used if USE_LOCAL_MIDI=True


    MAX_MIDI_FILES = 15000         # limit subset for conversion
    SEQ_LEN = 512
    TRAIN_TOKENS_TARGET = 100_000_000  # 100M tokens/model
    BATCH_TOKENS = 32_000       # adjust if OOM (e.g., 24_000)
    VAL_FRAC = 0.01
    TEST_FRAC = 0.01
    LR = 3e-4
    BETAS = (0.9, 0.95)
    WEIGHT_DECAY = 0.1
    CLIP_NORM = 1.0
    SAVE_DIR = "./runs_lakh"
    SEED = 42

cfg = Cfg()
os.makedirs(cfg.DATA_ROOT, exist_ok=True)
os.makedirs(cfg.MIDI_ROOT, exist_ok=True)
os.makedirs(cfg.ABC_ROOT, exist_ok=True)
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---------------------- Step 1: Download MIDI ZIP ----------------------

def get_lakh_subset_zip():
    zip_path = cfg.LMD_ZIP_PATH
    if not os.path.exists(zip_path):
        raise RuntimeError(f"ZIP not found at {zip_path}. Upload/scp it first.")
    print("Using local Lakh subset ZIP:", zip_path)
    return zip_path

def extract_midi_zip(zip_path):
    print("Extracting MIDI from ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            if member.filename.lower().endswith(".mid") or member.filename.lower().endswith(".midi"):
                zf.extract(member, cfg.MIDI_ROOT)
    print("Extracted MIDI into:", cfg.MIDI_ROOT)

# ---------------------- Step 2: MIDI -> Custom Text ----------------------

def midi_to_text(pm: pretty_midi.PrettyMIDI) -> str:
    """
    Convert a PrettyMIDI object into a simple line-based event format.
    Each note becomes one line:
      NOTE inst=<program> drum=<0/1> pitch=<int> start=<sec> dur=<sec> vel=<int>
    """
    lines = []
    for inst in pm.instruments:
        prog = inst.program
        is_drum = int(inst.is_drum)
        for n in inst.notes:
            start = float(n.start)
            dur = float(n.end - n.start)
            lines.append(
                f"NOTE inst={prog} drum={is_drum} pitch={n.pitch} "
                f"start={start:.3f} dur={dur:.3f} vel={n.velocity}"
            )
    return "\n".join(lines)

def convert_midi_to_abc():
    """
    Convert MIDI -> custom text representation using pretty_midi.
    Output files are stored in cfg.ABC_ROOT with .txt extension.
    """
    midi_files = glob.glob(os.path.join(cfg.MIDI_ROOT, "**", "*.mid"), recursive=True)
    print("Total MIDI files found:", len(midi_files))
    midi_files = midi_files[:cfg.MAX_MIDI_FILES]
    print("Converting first", len(midi_files), "files to custom text")

    if not midi_files:
        print("No MIDI files to convert.")
        return

    for midi_path in tqdm(midi_files, desc="MIDI->TEXT"):
        try:
            base = os.path.splitext(os.path.basename(midi_path))[0]
            out_path = os.path.join(cfg.ABC_ROOT, base + ".txt")
            if os.path.exists(out_path):
                continue  # skip already converted
            pm = pretty_midi.PrettyMIDI(midi_path)
            txt = midi_to_text(pm)
            if not txt.strip():
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(txt)
        except Exception as e:
            print("Failed:", midi_path, "->", e)
            continue
    print("Custom text files written to:", cfg.ABC_ROOT)

# ---------------------- Step 3: Load/clean Text ----------------------

def load_abc_texts():
    """
    Load the custom text notation files produced from MIDI.
    Function name kept for compatibility with the rest of the pipeline.
    """
    texts = []
    invalid = short = long_ = 0
    MIN_LEN, MAX_LEN = 50, 50_000
    txt_files = list(Path(cfg.ABC_ROOT).glob("*.txt"))
    print("Found text files:", len(txt_files))

    for f in tqdm(txt_files, desc="Loading text"):
        try:
            s = f.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            invalid += 1
            continue
        if len(s) < MIN_LEN:
            short += 1
            continue
        if len(s) > MAX_LEN:
            long_ += 1
            continue
        texts.append(s)

    if not texts:
        raise RuntimeError("No valid text sequences loaded.")

    lens = [len(t) for t in texts]
    total_chars = sum(lens)
    print(f"Valid sequences: {len(texts)}")
    print(f"invalid: {invalid}, too_short: {short}, too_long: {long_}")
    print(f"Total chars ≈ tokens: {total_chars:,}")
    print(f"len min/mean/median/max: {min(lens)}, {np.mean(lens):.1f}, {np.median(lens):.1f}, {max(lens)}")
    return texts

# ---------------------- Tokenizer ----------------------

class CharTokenizer:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.itos = ["<pad>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
    def encode(self, s):
        return [self.bos_id] + [self.stoi.get(ch, self.pad_id) for ch in s] + [self.eos_id]
    def decode(self, ids):
        rev = {i:c for i,c in enumerate(self.itos)}
        return "".join(rev[i] for i in ids if i >= 3)

# ---------------------- Streaming Dataset ----------------------

class ABCStream(IterableDataset):
    def __init__(self, texts, tokenizer, split):
        n = len(texts)
        val_n = int(n * cfg.VAL_FRAC)
        test_n = int(n * cfg.TEST_FRAC)
        train_n = n - val_n - test_n
        self.ranges = {
            "train": (0, train_n),
            "val":   (train_n, train_n + val_n),
            "test":  (train_n + val_n, n),
        }
        self.texts = texts
        self.tokenizer = tokenizer
        self.split = split

    def __iter__(self):
        lo, hi = self.ranges[self.split]
        split_texts = self.texts[lo:hi]

        if self.split == "train":
            tokens_generated = 0
            cycle = 0
            while tokens_generated < cfg.TRAIN_TOKENS_TARGET:
                cycle += 1
                idxs = list(range(len(split_texts)))
                random.shuffle(idxs)
                buf = []
                for i in idxs:
                    ids = self.tokenizer.encode(split_texts[i])
                    buf.extend(ids)
                    tokens_generated += len(ids)
                    while len(buf) >= cfg.SEQ_LEN + 1:
                        x = buf[:cfg.SEQ_LEN]
                        y = buf[1:cfg.SEQ_LEN + 1]
                        buf = buf[cfg.SEQ_LEN:]
                        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                    if tokens_generated >= cfg.TRAIN_TOKENS_TARGET:
                        break
                if cycle in (1,5,10):
                    print(f"  Completed cycle {cycle}, ~{tokens_generated:,} tokens generated")
        else:
            idxs = list(range(len(split_texts)))
            random.shuffle(idxs)
            buf = []
            for i in idxs:
                ids = self.tokenizer.encode(split_texts[i])
                buf.extend(ids)
                while len(buf) >= cfg.SEQ_LEN + 1:
                    x = buf[:cfg.SEQ_LEN]
                    y = buf[1:cfg.SEQ_LEN + 1]
                    buf = buf[cfg.SEQ_LEN:]
                    yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def make_loader(texts, tokenizer, split):
    batch = cfg.BATCH_TOKENS // cfg.SEQ_LEN
    print(f"{split} loader batch sequences:", batch)
    return DataLoader(
        ABCStream(texts, tokenizer, split),
        batch_size=batch,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

# ---------------------- Models ----------------------

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_p=0.1, ff_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_p, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(ff_p),
            nn.Linear(d_ff, d_model), nn.Dropout(ff_p),
        )
    def forward(self, x, mask):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h, attn_mask=mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(cfg.SEQ_LEN, d_model)
        self.blocks = nn.ModuleList(
            GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(cfg.SEQ_LEN, cfg.SEQ_LEN) * float("-inf"), 1))
    def forward(self, x):
        B,T = x.size()
        h = self.emb(x) + self.pos.weight[:T]
        mask = self.mask[:T,:T]
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h)
        return self.head(h)

class MiniLSTM(nn.Module):
    def __init__(self, vocab, d_model, n_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab)
    def forward(self, x):
        h = self.emb(x)
        out,_ = self.lstm(h)
        return self.head(out)

def tcfgs():
    return [
        ("tiny",   {"d_model":128,"n_layers":2,  "n_heads":4,  "d_ff":512}),
        ("small",  {"d_model":192,"n_layers":3,  "n_heads":4,  "d_ff":768}),
        ("medium", {"d_model":256,"n_layers":4,  "n_heads":4,  "d_ff":1024}),
        ("large",  {"d_model":320,"n_layers":8,  "n_heads":5,  "d_ff":1280}),
        ("xl",     {"d_model":768,"n_layers":16, "n_heads":12, "d_ff":3072}),  # ~100M+
    ]

def lcfgs():
    return [
        ("tiny_rnn",  {"d_model":128,"n_layers":1}),
        ("small_rnn", {"d_model":192,"n_layers":1}),
        ("med_rnn",   {"d_model":256,"n_layers":2}),
        ("large_rnn", {"d_model":320,"n_layers":3}),  # larger RNN to approach GPT sizes
    ]

# ---------------------- Training Utils ----------------------

def count_params(m): return sum(p.numel() for p in m.parameters())

def train_one_epoch(model, optimizer, train_loader, val_loader, track_curve=False):
    scaler = torch.amp.GradScaler("cuda") if device=="cuda" else torch.amp.GradScaler("cpu")
    t0 = time.time()
    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    total_loss = total_tokens = 0
    curve = [] if track_curve else None
    log_every = 100

    for step, (xb, yb) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_NORM)
        scaler.step(optimizer); scaler.update()
        tokens = xb.numel()
        total_loss += loss.item()*tokens
        total_tokens += tokens
        if track_curve and (step % log_every == 0):
            curve.append((step, loss.item()))
    train_loss = total_loss/total_tokens

    model.eval()
    val_loss = val_tokens = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="Val", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            tokens = xb.numel()
            val_loss += loss.item()*tokens
            val_tokens += tokens
    val_loss = val_loss/val_tokens
    wall = time.time()-t0
    mem_peak = torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else 0.0
    return train_loss, val_loss, wall, curve, mem_peak

# ---------------------- Scaling Runs ----------------------

def run_gpt_family(train_loader, val_loader):
    results = []
    curves  = {}
    for name, cfgm in tcfgs():
        if device=="cuda":
            torch.cuda.empty_cache()
        model = MiniGPT(vocab_size, **cfgm).to(device)
        params = count_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, betas=cfg.BETAS, weight_decay=cfg.WEIGHT_DECAY)
        print(f"\n{'='*60}\nTraining GPT: {name} ({params/1e6:.2f}M params)\n{'='*60}")
        tr,vl,wall,curve,mem = train_one_epoch(model, opt, train_loader, val_loader, track_curve=True)
        print(f"✓ {name}: train={tr:.3f} val={vl:.3f} time={wall/60:.1f}m mem={mem:.2f}GB")
        results.append({"model":name,"kind":"gpt","params":params,
                        "train_loss":tr,"val_loss":vl,"time_s":wall,"gpu_memory_gb":mem})
        if curve:
            curves[f"gpt_{name}"] = curve
        torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"gpt_{name}.pt"))
        del model, opt
    return results, curves

def run_lstm_family(train_loader, val_loader, results, curves):
    for name, cfgm in lcfgs():
        if device=="cuda":
            torch.cuda.empty_cache()
        model = MiniLSTM(vocab_size, **cfgm).to(device)
        params = count_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, betas=cfg.BETAS, weight_decay=cfg.WEIGHT_DECAY)
        print(f"\n{'='*60}\nTraining LSTM: {name} ({params/1e6:.2f}M params)\n{'='*60}")
        tr,vl,wall,curve,mem = train_one_epoch(model, opt, train_loader, val_loader, track_curve=True)
        print(f"✓ {name}: train={tr:.3f} val={vl:.3f} time={wall/60:.1f}m mem={mem:.2f}GB")
        results.append({"model":name,"kind":"lstm","params":params,
                        "train_loss":tr,"val_loss":vl,"time_s":wall,"gpu_memory_gb":mem})
        if curve:
            curves[f"lstm_{name}"] = curve
        torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, f"lstm_{name}.pt"))
        del model, opt
    return results, curves

# ---------------------- Scaling Plots ----------------------

def fit_power_law(xs, ys):
    c = min(ys)
    y_adj = np.array(ys) - c + 1e-8
    X = np.log(np.array(xs))
    Y = np.log(y_adj)
    A = np.vstack([X, np.ones_like(X)]).T
    alpha, loga = np.linalg.lstsq(A, Y, rcond=None)[0]
    return np.exp(loga), -alpha, c

def plot_scaling(results, kind, save_prefix):
    xs = [r["params"] for r in results if r["kind"]==kind]
    ys = [r["val_loss"] for r in results if r["kind"]==kind]
    if not xs:
        print("No results for kind:", kind)
        return None
    a, alpha, c = fit_power_law(xs, ys)
    xp = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
    yp = a * xp**(-alpha) + c
    plt.figure(figsize=(6,5))
    plt.scatter(xs, ys, s=80, label=f"{kind.upper()} val loss")
    plt.plot(xp, yp, "--", label=f"fit α={alpha:.3f}")
    plt.xscale("log")
    plt.xlabel("params")
    plt.ylabel("val loss")
    plt.title(f"{kind.upper()} scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(cfg.SAVE_DIR, f"{save_prefix}_{kind}_scaling.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved scaling plot for {kind} to", out)
    return alpha

def plot_combined(results):
    plt.figure(figsize=(7,5))
    for kind, marker, color in [("gpt","o","blue"),("lstm","s","red")]:
        xs = [r["params"] for r in results if r["kind"]==kind]
        ys = [r["val_loss"] for r in results if r["kind"]==kind]
        if xs:
            plt.scatter(xs, ys, marker=marker, s=80, label=kind.upper(), color=color)
    plt.xscale("log")
    plt.xlabel("params")
    plt.ylabel("val loss")
    plt.title("GPT vs LSTM scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(cfg.SAVE_DIR, "combined_scaling.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("Saved combined scaling plot to", out)

# ---------------------- Best Model Sampling & Eval ----------------------

def load_best(results, kind="gpt"):
    best = min([r for r in results if r["kind"]==kind], key=lambda x:x["val_loss"])
    name = best["model"]
    if kind=="gpt":
        cfgm = dict([c for c in tcfgs() if c[0]==name][0][1])
        model = MiniGPT(vocab_size, **cfgm).to(device)
    else:
        cfgm = dict([c for c in lcfgs() if c[0]==name][0][1])
        model = MiniLSTM(vocab_size, **cfgm).to(device)
    state = torch.load(os.path.join(cfg.SAVE_DIR, f"{kind}_{name}.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, best

def sample(model, kind, tokenizer, prompt="", max_new=200, temp=1.0, topk=20):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(x)[:,-1,:]
            logits = logits / temp
            if topk:
                v, idx = torch.topk(logits, k=min(topk, logits.size(-1)))
                logits = torch.full_like(logits, -1e9).scatter(1, idx, v)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)
        if x.size(1) >= cfg.SEQ_LEN:
            break
    return tokenizer.decode(x[0].tolist())

def abc_to_midi(abc_text, out_path):
    try:
        s = converter.parse(abc_text, format="abc")
        mf = m21_midi.translate.streamToMidiFile(s)
        mf.open(out_path, "wb")
        mf.write()
        mf.close()
        return True
    except Exception as e:
        print("Conversion error:", e)
        return False

def eval_best_model(texts, tokenizer, results, train_loader, val_loader, test_loader):
    # Test perplexity
    best_model, best_meta = load_best(results, "gpt")
    print("Best GPT model:", best_meta)

    def eval_ppl(loader, name):
        best_model.eval()
        total_loss = total_tokens = 0
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=name, leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = best_model(xb)
                loss = nn.functional.cross_entropy(logits.view(-1,vocab_size), yb.view(-1))
                tokens = xb.numel()
                total_loss += loss.item()*tokens
                total_tokens += tokens
        avg = total_loss/total_tokens
        return avg, math.exp(avg)

    test_loss, test_ppl = eval_ppl(test_loader, "Test")
    print(f"Test loss: {test_loss:.4f}, perplexity: {test_ppl:.2f}")

    # Generate 10 diverse samples
    os.makedirs(os.path.join(cfg.SAVE_DIR, "best_samples"), exist_ok=True)
    prompts = [
        "", "", "", "", "",
        "X:1\nT:Air\nM:3/4\nK:G\n",
        "X:1\nT:Reel\nM:4/4\nK:Dmix\n",
        "X:1\nT:Jig\nM:6/8\nK:Am\n",
        "X:1\nT:Waltz\nM:3/4\nK:C\n",
        "X:1\nT:Hornpipe\nM:4/4\nK:G\n",
    ]
    def looks_valid_abc(txt):
        return any(h in txt for h in ["X:","T:","M:","K:"])

    valid = ok_midi = 0
    meta = []
    for i, p in enumerate(prompts):
        abc_text = sample(best_model, "gpt", tokenizer, prompt=p, max_new=256, temp=0.9, topk=20)
        abc_path = os.path.join(cfg.SAVE_DIR, "best_samples", f"sample_{i+1}.abc")
        midi_path = os.path.join(cfg.SAVE_DIR, "best_samples", f"sample_{i+1}.mid")
        Path(abc_path).write_text(abc_text, encoding="utf-8")
        is_valid = looks_valid_abc(abc_text)
        valid += int(is_valid)
        converted = abc_to_midi(abc_text, midi_path)
        ok_midi += int(converted)
        meta.append({"idx":i+1,"prompt":p,"abc":abc_path,"midi":midi_path,
                     "valid_abc":bool(is_valid),"midi_ok":bool(converted)})
    valid_pct = 100*valid/len(prompts)
    midi_pct = 100*ok_midi/len(prompts)
    print(f"Sample validity: {valid}/{len(prompts)} ({valid_pct:.1f}%)")
    print(f"MIDI success:    {ok_midi}/{len(prompts)} ({midi_pct:.1f}%)")
    json.dump(meta, open(os.path.join(cfg.SAVE_DIR, "best_samples", "metadata.json"),"w"), indent=2)

# ---------------------- Main ----------------------

def main():
    # 1) Download/extract MIDI or use local MIDI
    if cfg.USE_LOCAL_MIDI:
        print("Using local MIDI root:", cfg.LOCAL_MIDI_ROOT)
        cfg.MIDI_ROOT = cfg.LOCAL_MIDI_ROOT
    else:
        zip_path = get_lakh_subset_zip()
        extract_midi_zip(zip_path)

    # 2) Convert MIDI -> ABC (idempotent if already run)
    convert_midi_to_abc()

    # 3) Load ABC texts
    texts = load_abc_texts()
    random.seed(cfg.SEED)
    random.shuffle(texts)

    # 4) Tokenizer
    tokenizer = CharTokenizer(texts)
    vocab_size_local = len(tokenizer.itos)
    global vocab_size
    vocab_size = vocab_size_local
    print("Tokenizer vocab_size:", vocab_size)

    # 5) Loaders (98/1/1 split; 100M tokens via ABCStream)
    train_loader = make_loader(texts, tokenizer, "train")
    val_loader   = make_loader(texts, tokenizer, "val")
    test_loader  = make_loader(texts, tokenizer, "test")

    # 6) Train GPT + LSTM families
    results, curves = run_gpt_family(train_loader, val_loader)
    results, curves = run_lstm_family(train_loader, val_loader, results, curves)

    json.dump(results, open(os.path.join(cfg.SAVE_DIR, "scaling_results.json"),"w"), indent=2)
    json.dump(curves,  open(os.path.join(cfg.SAVE_DIR, "training_curves.json"),"w"), indent=2)

    # 7) Scaling plots + power-law fits
    alpha_gpt  = plot_scaling(results, "gpt",  "gpt")
    alpha_lstm = plot_scaling(results, "lstm", "lstm")
    plot_combined(results)
    print(f"Transformer scaling exponent α = {alpha_gpt}")
    print(f"LSTM scaling exponent α = {alpha_lstm}")

    # 8) Best-model eval and samples
    eval_best_model(texts, tokenizer, results, train_loader, val_loader, test_loader)
    print("All done. Outputs in:", cfg.SAVE_DIR)

if __name__ == "__main__":
    main()