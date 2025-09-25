import argparse, torch, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    regs = torch.load(args.path, map_location="cpu")
    plt.figure()
    plt.plot(regs.numpy(), label="TS (vMF–MH)")
    plt.xlabel("t"); plt.ylabel("Expected regret")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
