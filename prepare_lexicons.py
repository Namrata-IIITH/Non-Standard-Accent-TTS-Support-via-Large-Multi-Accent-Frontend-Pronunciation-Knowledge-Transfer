import re
import os
import pickle

ACCENT_MAP = {
    "abd1": "lex_abd1_23116_3unique.unknown",
    "abc":  "lex_abc_12560_3unique.unknown",
    "ccl1": "lex_ccl1_20144_3unique.unknown",
    "cdf":  "lex_cdf_5460_3unique.unknown",
    "edi":  "lex_edi_15264_3unique.unknown",
    "gam":  "lex_gam_13992_3unique.unknown",
    "gau":  "lex_gau_10828_3unique.unknown",
    "gnz":  "lex_gnz_26232_3unique.unknown",
    "lds":  "lex_lds_21016_3unique.unknown",
    "lds1": "lex_lds1_17728_3unique.unknown",
    "nyc":  "lex_nyc_15080_3unique.unknown",
    "nyc1": "lex_nyc1_18824_3unique.unknown",
    "rpx":  "lex_rpx_18576_3unique.unknown",
    "sca":  "lex_sca_15496_3unique.unknown"
}

LEX_DIR = "/media/nayan/g/harshitha/interns/namrata/lexicons"

def parse_lexicon(path):
    lexicon = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            m = re.search(r'^([^:]+).*?\{(.*?)\}', line)
            if not m:
                continue

            word = m.group(1).lower()
            toks = m.group(2).split()

            stress = False
            phones = []

            for t in toks:
                if t == "*":
                    stress = True
                elif t == ".":
                    phones.append("-")
                elif t in ["<",">","{","}"]:
                    continue
                else:
                    s = "1" if stress else "0"
                    phones.append(t + s)
                    stress = False

            phones.append("+")
            lexicon[word] = " ".join(phones)

    return lexicon


if __name__ == "__main__":
    os.makedirs("parsed_lexicons", exist_ok=True)

    for acc, fname in ACCENT_MAP.items():
        path = os.path.join(LEX_DIR, fname)

        if not os.path.exists(path):
            print("MISSING:", path)
            continue

        print("Parsing:", fname)
        lex = parse_lexicon(path)

        with open(f"parsed_lexicons/{acc}.pkl", "wb") as f:
            pickle.dump(lex, f)

    print("DONE: saved 14 lexicons into parsed_lexicons/")
