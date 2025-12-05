import argparse
import sys

def load_vocab(train_path):
    seen = set()
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    seen.add(word)
    except FileNotFoundError:
        print(f"Error: Could not find training file at {train_path}")
        sys.exit(1)
    return seen

def extract_boundaries(phone_seq):
    return [token for token in phone_seq.split() if token in ['+', 'B', 'BB', '$']]

def calculate_metrics(pred_path, tgt_path, src_path, seen_vocab):
    with open(pred_path, 'r') as f_p, open(tgt_path, 'r') as f_t, open(src_path, 'r') as f_s:
        preds = [l.strip() for l in f_p]
        truths = [l.strip() for l in f_t]
        sources = [l.strip() for l in f_s]

    bound_correct = 0
    bound_total = 0
    
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0

    for p_sent, t_sent, s_text in zip(preds, truths, sources):
        p_bounds = extract_boundaries(p_sent)
        t_bounds = extract_boundaries(t_sent)
        
        if p_bounds == t_bounds:
            bound_correct += 1
        bound_total += 1
        
        s_words = s_text.split()
        
        p_word_phones = p_sent.split('+')
        t_word_phones = t_sent.split('+')

        if len(p_word_phones) != len(t_word_phones):
            for w in s_words:
                if w in seen_vocab: 
                    seen_total += 1
                else: 
                    unseen_total += 1
            continue

        for i, word in enumerate(s_words):
            if i >= len(p_word_phones): break
            
            p_clean = p_word_phones[i].strip()
            t_clean = t_word_phones[i].strip()
            
            is_correct = (p_clean == t_clean)
            
            if word in seen_vocab:
                seen_total += 1
                if is_correct: seen_correct += 1
            else:
                unseen_total += 1
                if is_correct: unseen_correct += 1
    
    
    print(f"{'METRIC':<30} | {'ACCURACY':<10} | {'COUNTS'}")
    
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

    vocab = load_vocab(args.train_src)
    
    calculate_metrics(args.pred_tgt, args.test_tgt, args.test_src, vocab)