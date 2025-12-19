from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MVTEC = ROOT / "data" / "mvtec_ad"

REQUIRED = [
    ("train", "good"),
    ("test", "good"),
]

def main():
    print(f"[i] project root: {ROOT}")
    print(f"[i] expected mvtec path: {MVTEC}")

    if not MVTEC.exists():
        print(f"[X] Not found: {MVTEC}")
        print("→ Create: data/mvtec_ad and put category folders (bottle, cable, ...) inside it.")
        return

    cats = [p for p in MVTEC.iterdir() if p.is_dir()]
    print(f"[i] category folders found: {len(cats)}")
    if not cats:
        print("[X] No category folders inside mvtec_ad.")
        print("→ mvtec_ad should directly contain folders like: bottle/, cable/, capsule/, ...")
        return

    ok = 0
    for c in sorted(cats):
        missing = []
        for a, b in REQUIRED:
            if not (c / a / b).exists():
                missing.append(f"{a}/{b}")
        if missing:
            print(f"[X] {c.name}: missing {missing}")
        else:
            print(f"[OK] {c.name}")
            ok += 1

    print(f"\nDone. OK categories: {ok}/{len(cats)}")

if __name__ == "__main__":
    main()
