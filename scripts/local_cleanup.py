"""Safely archive obsolete root-level variants without deleting them.

Default is PREVIEW mode. --apply moves files into ignored archive_local/.
Execute from the repository root after backing up your original directory.
"""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parent.parent
OBSOLETE = (
    'api_backup.py', 'apiday9.py', 'claim_evidence_review old.py',
    'agent_day3.py', 'agent_citation_audit.py', 'claim_evidence_review.py',
    'research_agent_audited.py', 'research_agent_claim_review.py',
    'review_saved_claims.py', 'smoke_test_day10.py',
    'test_agent_citation_audit.py', 'test_claim_evidence_review.py',
    'test_research_agent.py', 'data/eval_questions11.json',
    'README_AGENT.md',
)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Move legacy files to archive_local/')
    args = ap.parse_args()
    for item in OBSOLETE:
        src = ROOT / item
        if not src.is_file():
            continue
        dest = ROOT / 'archive_local' / item
        if dest.exists():
            print('[SKIP] destination already exists:', dest)
            continue
        if args.apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            print('[MOVED]', item)
        else:
            print('[PREVIEW]', item, '->', dest.relative_to(ROOT))
    if not args.apply:
        print('Preview only. Re-run with --apply after backing up.')
if __name__ == '__main__':
    main()
