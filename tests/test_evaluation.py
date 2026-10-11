"""Metrics-only tests: no torch/PDF/API required."""
from src.evaluation import evaluate, verified_questions

def run_tests():
    rows = [
        {'query':'q1','relevant':[{'source':'a.pdf','page':3}], 'label_status':'source_page_evidence_verified'},
        {'query':'q2','relevant':[], 'label_status':'pending'},
    ]
    assert len(verified_questions(rows)) == 1
    assert evaluate([{'source':'b.pdf','page':2},{'source':'a.pdf','page':3}], rows[0]['relevant']) == (1.0, 0.5)
    assert evaluate([{'source':'b.pdf','page':2}], rows[0]['relevant']) == (0.0, 0.0)
    try:
        evaluate([], [])
        raise AssertionError('Empty labels should fail')
    except ValueError:
        pass
    print('[PASS] verified-only scoring and MRR metric')

if __name__ == '__main__':
    run_tests()
