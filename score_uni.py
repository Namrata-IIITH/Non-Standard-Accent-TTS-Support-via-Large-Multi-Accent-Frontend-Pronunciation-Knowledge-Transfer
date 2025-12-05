import argparse
import sys

def load_vocab(train_path):
    """Builds a set of all words seen during training."""
    seen = set()
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming source text is space-separated words
                for word in line.strip().split():
                    seen.add(word)
    except FileNotFoundError:
        print(f"Error: Could not find training file at {train_path}")
        sys.exit(1)
    return seen

def extract_boundaries(phone_seq):
    """
    Extracts boundary symbols for the 'Boundary Accuracy' metric.
    Includes Word boundaries (+), Prosodic boundaries (B, BB), etc.
    """
    return [token for token in phone_seq.split() if token in ['+', 'B', 'BB', '$']]

def calculate_metrics(pred_path, tgt_path, src_path, seen_vocab):
    with open(pred_path, 'r') as f_p, open(tgt_path, 'r') as f_t, open(src_path, 'r') as f_s:
        preds = [l.strip() for l in f_p]
        truths = [l.strip() for l in f_t]
        sources = [l.strip() for l in f_s]

    # Metric Counters
    bound_correct = 0
    bound_total = 0
    
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0

    for p_sent, t_sent, s_text in zip(preds, truths, sources):
        # --- 1. Boundary Accuracy ---
        # "Word boundary & prosodic boundary label accuracy rates" [cite: 159]
        p_bounds = extract_boundaries(p_sent)
        t_bounds = extract_boundaries(t_sent)
        
        # Strict matching for the sequence of boundaries
        if p_bounds == t_bounds:
            bound_correct += 1
        bound_total += 1

        # --- 2 & 3. Word Accuracy (Seen vs Unseen) ---
        # "Word accuracy rates... consider phone symbols, lexical stress labels, 
        # and syllable break labels".
        
        s_words = s_text.split()
        
        # Split phones by word boundary '+' to get per-word pronunciations
        p_word_phones = p_sent.split('+')
        t_word_phones = t_sent.split('+')

        # If model prediction has wrong number of words (missing/extra '+'), 
        # we cannot align words perfectly. Count all as wrong.
        if len(p_word_phones) != len(t_word_phones):
            # If lengths differ, we assume all words in this sentence are "wrong"
            # just to keep the counters accurate for the total denominator.
            for w in s_words:
                if w in seen_vocab: 
                    seen_total += 1
                else: 
                    unseen_total += 1
            continue

        # Check each word individually
        for i, word in enumerate(s_words):
            if i >= len(p_word_phones): break
            
            # --- CRITICAL FIX FOR EXPERIMENT 4.2 ---
            # Do NOT remove syllable breaks ('-').
            # The metric requires checking phone + stress + breaks.
            p_clean = p_word_phones[i].strip()
            t_clean = t_word_phones[i].strip()
            
            # Strict string comparison
            is_correct = (p_clean == t_clean)
            
            if word in seen_vocab:
                seen_total += 1
                if is_correct: seen_correct += 1
            else:
                unseen_total += 1
                if is_correct: unseen_correct += 1
    
    # Sanity check for Experiment 4.2
    if len(seen_vocab) > 10000:
        print("WARNING: High vocab count detected. Are you pointing to the '1k' dataset?")
        print("         For Exp 4.2, --train_src must be 'dataset_1k/unilex_edi/src-train.txt'")
    
    print("-" * 80)
    print(f"{'METRIC':<30} | {'ACCURACY':<10} | {'COUNTS'}")
    print("-" * 80)
    
    b_acc = bound_correct/bound_total if bound_total > 0 else 0
    print(f"{'1. Boundary Accuracy':<30} | {b_acc:.2%}     | ({bound_correct}/{bound_total})")
    
    s_acc = seen_correct/seen_total if seen_total > 0 else 0
    print(f"{'2. Seen Word Accuracy':<30} | {s_acc:.2%}     | ({seen_correct}/{seen_total})")
    
    u_acc = unseen_correct/unseen_total if unseen_total > 0 else 0
    print(f"{'3. Unseen Word Accuracy':<30} | {u_acc:.2%}     | ({unseen_correct}/{unseen_total})")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", required=True, help="Path to training source text (defines 'Seen' words)")
    parser.add_argument("--test_src", required=True, help="Source text for test set")
    parser.add_argument("--test_tgt", required=True, help="Ground truth phonemes for test set")
    parser.add_argument("--pred_tgt", required=True, help="Model predictions")
    args = parser.parse_args()

    # Load the training vocabulary to define 'Seen'
    vocab = load_vocab(args.train_src)
    
    # Run calculation
    calculate_metrics(args.pred_tgt, args.test_tgt, args.test_src, vocab)