import os
import random
import shutil

INPUT_ROOT = "dataset"          
OUTPUT_ROOT = "dataset_1k"      
# TARGET_ACCENT = "unilex_edi"    
SOURCE_ACCENT = "unilex_gnz"    
TARGET_SIZE = 1000              

random.seed(42)

vocab_file = "./checkpoints/large_multi_baseline/tc_multi_14_accents-December-02-2025_12+22PM-ee3e8e4/vocab/tgt.vocab"
KEEP_SYMBOLS = {'+', '-', 'B', 'BB', '$'} 

def load_vocab_set(path):
    v = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts: v.add(parts[0])
    except FileNotFoundError:
        print(f"ERROR: Vocab file not found at {path}"); exit(1)
    return v

valid_phones = load_vocab_set(vocab_file)

def clean_and_validate(src_list, tgt_list):
    final_src = []
    final_tgt = []
    
    for s, t in zip(src_list, tgt_list):
        phones = t.strip().split()
        filtered = [ph for ph in phones if (ph in valid_phones or ph in KEEP_SYMBOLS)]
        clean_t = " ".join(filtered)
        
        if len(clean_t) > 0:
            final_src.append(s)
            final_tgt.append(clean_t + "\n")
            
    return final_src, final_tgt

def load_train_lines(accent):
    src = open(os.path.join(INPUT_ROOT, accent, "src-train.txt")).readlines()
    tgt = open(os.path.join(INPUT_ROOT, accent, "tgt-train.txt")).readlines()
    assert len(src) == len(tgt), "src/tgt mismatch"
    return src, tgt

def load_test_lines(accent):
    src = open(os.path.join(INPUT_ROOT, accent, "src-test.txt")).readlines()
    tgt = open(os.path.join(INPUT_ROOT, accent, "tgt-test.txt")).readlines()
    return src, tgt

edi_src, edi_tgt = load_train_lines(TARGET_ACCENT)
edi_indices = list(range(len(edi_src)))
random.shuffle(edi_indices)
edi_indices = edi_indices[:TARGET_SIZE]

print(f"[OK] Selected {TARGET_SIZE} training lines for EDI.")

if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT)

accents = sorted([d for d in os.listdir(INPUT_ROOT) if d.startswith("unilex_")])

for acc in accents:
    print(f"Processing {acc}...")
    in_dir = os.path.join(INPUT_ROOT, acc)
    out_dir = os.path.join(OUTPUT_ROOT, acc)
    os.makedirs(out_dir, exist_ok=True)

    for split in ["test", "val"]:
        src_path = os.path.join(in_dir, f"src-{split}.txt")
        tgt_path = os.path.join(in_dir, f"tgt-{split}.txt")
        s_raw, t_raw = open(src_path).readlines(), open(tgt_path).readlines()
        
        s_clean, t_clean = clean_and_validate(s_raw, t_raw)
        
        with open(os.path.join(out_dir, f"src-{split}.txt"), "w") as f: f.writelines(s_clean)
        with open(os.path.join(out_dir, f"tgt-{split}.txt"), "w") as f: f.writelines(t_clean)

    src, tgt = load_train_lines(acc)
    N = len(src)
    raw_src, raw_tgt = [], []

    if acc == TARGET_ACCENT:
        print(" -> Target accent: using fixed 1k subset.")
        raw_src = [src[i] for i in edi_indices]
        raw_tgt = [tgt[i] for i in edi_indices]

    elif acc == SOURCE_ACCENT:
        print(" -> Source accent: 1k + Augmentation.")
        idx = list(range(N)); random.shuffle(idx); chosen = idx[:TARGET_SIZE]
        
        raw_src = [src[i] for i in chosen]
        raw_tgt = [tgt[i] for i in chosen]

        aug_src, aug_tgt = load_test_lines(acc)
        
        raw_src += aug_src
        raw_tgt += aug_tgt
        print(f"    Added {len(aug_src)} augmentation lines from {acc} test set.")

    else:
        available_indices = list(set(range(N)) - set(edi_indices))
        random.shuffle(available_indices)
        chosen = available_indices[:TARGET_SIZE]
        
        raw_src = [src[i] for i in chosen]
        raw_tgt = [tgt[i] for i in chosen]

    final_src, final_tgt = clean_and_validate(raw_src, raw_tgt)
    
    with open(os.path.join(out_dir, "src-train.txt"), "w") as f: f.writelines(final_src)
    with open(os.path.join(out_dir, "tgt-train.txt"), "w") as f: f.writelines(final_tgt)

print("\n Experiment 4.3 dataset_1k ")