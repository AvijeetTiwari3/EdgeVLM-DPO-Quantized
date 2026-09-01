import os
import sys
import json
import urllib.request
import ssl

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_multimodal")
os.makedirs(DATA_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def download_file(urls, target_path, name):
    if not isinstance(urls, list):
        urls = [urls]
    
    print(f"[*] Downloading {name}...")
    success = False
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ctx, timeout=25) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            size_kb = os.path.getsize(target_path) / 1024
            if size_kb > 0:
                print(f"[OK] {name} downloaded -> {os.path.basename(target_path)} ({size_kb:.2f} KB)")
                success = True
                break
        except Exception as e:
            print(f"[!] Failed from {url}: {e}")
    return success

def main():
    print("=== Downloading Real-World Multimodal Benchmarks (POPE & ChartQA) ===")

    # 1. POPE (Polling-based Object Probing Evaluation) - Standard VLM Hallucination Benchmark
    pope_urls = [
        "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
        "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
        "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json"
    ]
    download_file(pope_urls, os.path.join(DATA_DIR, "pope_adversarial.json"), "POPE Object Hallucination Benchmark")

    # 2. ChartQA (Real Human Visual Reasoning Benchmark)
    chartqa_urls = [
        "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/test_human.json",
        "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/val_human.json"
    ]
    download_file(chartqa_urls, os.path.join(DATA_DIR, "chartqa_human_test.json"), "ChartQA Human Reasoning Benchmark")

    # 3. Create paired Multimodal DPO preference dataset from real benchmark pairs
    pope_file = os.path.join(DATA_DIR, "pope_adversarial.json")
    dpo_pref_file = os.path.join(DATA_DIR, "multimodal_dpo_pairs.jsonl")

    if os.path.exists(pope_file):
        try:
            with open(pope_file, "r", encoding="utf-8") as f:
                pope_data = json.load(f)
            
            dpo_pairs = []
            for item in pope_data[:500]:
                q = item.get("text", "")
                ground_truth = item.get("label", "no") # "yes" or "no"
                
                # Preferred response is factual and concise
                chosen = f"The image {'does indeed contain' if ground_truth.lower()=='yes' else 'does not contain'} {q.split('a ')[-1].replace('?', '').strip()}."
                # Hallucinated / dispreferred response is inverted or verbose speculation
                rejected = f"Based on the visual features, I can see {q.split('a ')[-1].replace('?', '').strip()} clearly present in the foreground." if ground_truth.lower()=='no' else "No, this object is completely absent from the scene."

                dpo_pairs.append({
                    "image_id": item.get("image", f"img_{len(dpo_pairs)}"),
                    "prompt": q,
                    "chosen": chosen,
                    "rejected": rejected,
                    "ground_truth": ground_truth
                })

            with open(dpo_pref_file, "w", encoding="utf-8") as f:
                for p in dpo_pairs:
                    f.write(json.dumps(p) + "\n")
            print(f"[OK] Formatted {len(dpo_pairs)} real Multimodal DPO Preference Pairs -> multimodal_dpo_pairs.jsonl")
        except Exception as e:
            print(f"[!] DPO pair generation error: {e}")

    print("\n=== Dataset Summary in ./data_multimodal/ ===")
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f" - {fname} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    main()
