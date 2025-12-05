import os
import pickle
import re

ACCENTS = [
    "abd1","abc","ccl1","cdf","edi","gam","gau","gnz",
    "lds","lds1","nyc","nyc1","rpx","sca"
]

SRC_DIR = "/media/nayan/g/harshitha/interns/namrata/lexicons"
PARSED_DIR = "parsed_lexicons"
OUT_ROOT = "dataset"

MASTER = {
    "train": os.path.join(SRC_DIR, "src-train.txt"),
    "val":   os.path.join(SRC_DIR, "src-val.txt"),
    "test":  os.path.join(SRC_DIR, "src-test.txt")
}

def process_one_accent(accent):
    os.makedirs(f"{OUT_ROOT}/unilex_{accent}", exist_ok=True)

    with open(f"{PARSED_DIR}/{accent}.pkl", "rb") as f:
        lex = pickle.load(f)

    for split, path in MASTER.items():
        out_src = f"{OUT_ROOT}/unilex_{accent}/src-{split}.txt"
        out_tgt = f"{OUT_ROOT}/unilex_{accent}/tgt-{split}.txt"

        count = 0
        with open(path, "r") as fin, \
             open(out_src, "w") as fs, \
             open(out_tgt, "w") as ft:

            for line in fin:
                text = line.strip()
                clean = re.sub(r"[^\w\s']", "", text.lower())
                words = clean.split()

                phones = []
                ok = True

                for w in words:
                    if w in lex:
                        phones.append(lex[w])
                    else:
                        ok = False
                        break

                if ok:
                    fs.write(text + "\n")
                    ft.write(" ".join(phones) + "\n")
                    count += 1

        print(accent, split, count, "sentences")

if __name__ == "__main__":
    for a in ACCENTS:
        print("\nAccent:", a)
        process_one_accent(a)

    print("\nAll datasets written under dataset/")
