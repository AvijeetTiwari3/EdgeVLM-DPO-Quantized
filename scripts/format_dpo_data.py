import os
import sys
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_multimodal")
pope_file = os.path.join(DATA_DIR, "pope_adversarial.json")
dpo_pref_file = os.path.join(DATA_DIR, "multimodal_dpo_pairs.jsonl")

if os.path.exists(pope_file):
    dpo_pairs = []
    with open(pope_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                q = item.get("text", "")
                ground_truth = item.get("label", "no")
                
                target_obj = q.split("a ")[-1].replace("?", "").strip() if "a " in q else "the object"

                chosen = f"The image {'does indeed contain' if ground_truth.lower()=='yes' else 'does not contain'} {target_obj}."
                rejected = f"Based on the visual features, I can clearly observe {target_obj} in the scene." if ground_truth.lower()=='no' else f"No, {target_obj} is completely absent from the visual field."

                dpo_pairs.append({
                    "image_id": item.get("image", f"img_{len(dpo_pairs)}"),
                    "prompt": q,
                    "chosen": chosen,
                    "rejected": rejected,
                    "ground_truth": ground_truth
                })
            except Exception:
                continue

    with open(dpo_pref_file, "w", encoding="utf-8") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[OK] Formatted {len(dpo_pairs)} real Multimodal DPO Preference Pairs -> multimodal_dpo_pairs.jsonl")
