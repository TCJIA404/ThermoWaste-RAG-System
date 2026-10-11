"""Conservative, offline TRIAGE of ThermoWaste Claim Evidence Review JSON.
No LLM/API/PDF access; does not certify scientific factual correctness.
Python 3.8+. Usage:
    python review_saved_claims.py outputs/claim_review_20261011_141809.json
"""
import argparse
import csv
import json
import re
from pathlib import Path

CITATION = re.compile(r'\(?[\w.-]+\.pdf\s*[,，]\s*p\.?\s*\d+\)?', re.I)
NUMBER = re.compile(r'(?<![\w.])(?<!H)(?<!CO)(?<!CH)([-−+]?\d+(?:\.\d+)?)(?![\w.])')


def strip_references(text):
    text = CITATION.sub('', text)
    return re.sub(r'\b(?:H|CO|CH|O|ER|CCE)\s*[₂₄0-9]*\b', '', text)


def numeric_terms(text):
    """Review prompts, not scientific measurement extraction.
    Keep only decimal numbers or integers >= 5 after source references are removed.
    """
    text = strip_references(text)
    terms = []
    for match in NUMBER.finditer(text):
        token = match.group(1).replace('−','-').lstrip('+')
        try:
            value = float(token)
        except ValueError:
            continue
        if '.' in token or abs(value) >= 5:
            if token not in terms:
                terms.append(token)
    return terms


def summarize_flags(claim, quote):
    flags = []
    issues = []
    raw = claim
    lower = raw.lower()
    if raw.count('；') + raw.count(';') >= 1 or raw.count('，') + raw.count(',') >= 2 or len(CITATION.findall(raw)) >= 2:
        flags.append('COMPOUND_CLAIM')
        issues.append('同一条包含多个断言，单个 SUPPORTED 标签不足以表示各断言全部成立')
    missing = [x for x in numeric_terms(raw) if x not in numeric_terms(quote)]
    if missing:
        flags.append('NUMBERS_NOT_SHOWN_IN_EVIDENCE_QUOTE')
        issues.append('短引文未覆盖全部数字: ' + ', '.join(missing) + '；需查原文/完整检索片段')
    if re.search(r'(?:高温|升高温度|提高气化温度|increas\w* temperatur\w*)[^。\n]{0,80}(?:水气变换|水煤气变换|water.gas shift|\bWGS\b)', raw, re.I):
        flags.append('WGS_TEMPERATURE_MECHANISM_REVIEW')
        issues.append('温度与正向水煤气变换的关系不能笼统写成“高温促进”，须区分平衡、动力学和耦合反应')
    if ('产氢' in raw or '氢气生产' in raw or 'hydrogen yield' in lower) and re.search(r'(?:H[₂2]|氢气)(?:.{0,7})(?:浓度|占比|体积分数)', raw, re.I):
        flags.append('H2_CONCENTRATION_VS_YIELD')
        issues.append('核对 H₂ 浓度、体积分数与绝对产氢量的指标是否被混用')
    if ('热值与能耗' in raw and 'LHV' in raw):
        flags.append('LHV_NOT_ENERGY_COST')
        issues.append('LHV 降低不能单独证明系统能耗增加；需要独立能耗证据')
    if ('焦油裂解' in raw and ('相关系数' in raw or '-0.48' in raw or '−0.48' in raw)):
        flags.append('TAR_CORRELATION_VS_CRACKING')
        issues.append('焦油相关负相关系数不能单独证明焦油裂解速率提高；核对统计指标和机理')
    if ('SHAP' in raw and ('相关系数' in raw or 'r =' in lower)):
        flags.append('SHAP_VS_CORRELATION')
        issues.append('相关系数和平均绝对 SHAP 是不同统计指标，核对每个数字的含义')
    if ('CO' in raw and 'H₂' in raw and ('800' in raw or '900' in raw)):
        flags.append('COMPARE_H2_AND_CO_OBJECTIVES')
        issues.append('CO 的较高区间不等于 H₂ 的最佳区间，核对操作条件及目标指标')
    if (any(name in raw for name in ('水气变换', '水煤气变换', '水煤气转换')) or 'water-gas shift' in lower or 'water gas shift' in lower or 'wgs' in lower) and ('吸热' in raw or 'endothermic' in lower) and not any(phrase in raw for phrase in ('并非吸热', '不是吸热', '非吸热')) and 'not endothermic' not in lower:
        flags.append('WGS_ENTHALPY')
        issues.append('正向水煤气变换为放热反应，不应归入吸热反应')
    if not flags:
        flags.append('UNVERIFIED_AGAINST_FULL_PDF')
        issues.append('目前只有摘录摘要，仍需对照完整 PDF 检查全部断言')
    return flags, issues, missing


def run(input_file):
    p = Path(input_file)
    doc = json.loads(p.read_text(encoding='utf-8'))
    rows = []
    for c in doc['claims']:
        lr = c.get('llm_review') or {}
        flags, issues, missing = summarize_flags(c['claim'], lr.get('evidence_quote', ''))
        rows.append({
            'id': c['id'],
            'claim': c['claim'],
            'original_llm_verdict': lr.get('verdict'),
            'citation_pages_present': c.get('all_cited_pages_returned'),
            'human_review': c.get('human_review', 'PENDING'),
            'triage_status': 'REQUIRES_MANUAL_EVIDENCE_CHECK',
            'flags': flags,
            'why': issues,
            'numeric_terms_missing_from_single_quote': missing,
        })
    out = {'source_report': p.name, 'note': 'Diagnostic checklist ONLY; does not assert that any claim is factually false or verified', 'claims': rows}
    json_path = p.with_name(p.stem + '_quality_gate.json')
    csv_path = p.with_name(p.stem + '_quality_gate.csv')
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'original_llm_verdict', 'triage_status', 'flags', 'issues'])
        for r in rows:
            w.writerow([r['id'], r['original_llm_verdict'], r['triage_status'], '; '.join(r['flags']), '；'.join(r['why'])])
    for r in rows:
        print('{} [{}] -> {} | {}'.format(r['id'], r['original_llm_verdict'], r['triage_status'], ', '.join(r['flags'])))
    print('Reports:', json_path, csv_path)
    return out

