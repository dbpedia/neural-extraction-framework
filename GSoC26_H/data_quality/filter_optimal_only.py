import json

INPUT_FILE = "/home/nsingh/exp1_train_combined.jsonl"
OUTPUT_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"

kept = 0
total = 0

with open(INPUT_FILE, encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        entry = json.loads(line)
        total += 1
        if entry.get("trace_type") == "optimal":
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

print(f"Total entries scanned: {total}")
print(f"Optimal-only entries written: {kept}")
print(f"Output: {OUTPUT_FILE}")
