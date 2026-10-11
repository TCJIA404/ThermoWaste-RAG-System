"""Review a saved claim report without running GLM (run from project root)."""
import argparse
from agent.quality_gate import run

def main():
    p = argparse.ArgumentParser(description="Offline claim quality diagnostics")
    p.add_argument("report", help="Local outputs/claim_review_*.json file")
    args = p.parse_args()
    run(args.report)

if __name__ == "__main__":
    main()
