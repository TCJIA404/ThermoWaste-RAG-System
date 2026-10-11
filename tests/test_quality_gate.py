"""Offline rule safety test (a flag is a warning, not a factual verdict)."""
from agent.quality_gate import summarize_flags

def run_tests():
    claim = '升高温度促进吸热的水煤气变换反应，合成气产率与温度相关 r=0.17。'
    flags, issues, missing = summarize_flags(claim, 'Temperature positively correlates with yield')
    assert 'WGS_ENTHALPY' in flags
    assert 'COMPOUND_CLAIM' in flags or 'WGS_TEMPERATURE_MECHANISM_REVIEW' in flags
    assert isinstance(issues, list) and isinstance(missing, list)
    flags3, *_ = summarize_flags('正向水煤气变换反应是放热的。', '')
    assert 'WGS_ENTHALPY' not in flags3
    flags2, *_ = summarize_flags('This evidence may be incomplete.', '')
    assert 'UNVERIFIED_AGAINST_FULL_PDF' in flags2
    print('[PASS] cautious quality gate for WGS and unverified assertions')

if __name__ == '__main__':
    run_tests()
