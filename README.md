# Redrob Intelligent Candidate Ranking System

**Hackathon:** Redrob Data & AI Challenge  
**Author:** Sirisha Gowda | [Sirishagowda2025](https://github.com/Sirishagowda2025)

## Quick Start

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

## Architecture

Three-stage hybrid scoring pipeline:

1. **Skill Semantics** — Scores *career descriptions*, not just skills section. Finds implicit expertise (a candidate who "built recommendation system at scale" matches even without saying "RAG").

2. **Career Fit** — YOE (6-8yr sweet spot), company type (product vs consulting penalized), title relevance, production deployment evidence in descriptions, location.

3. **Behavioral Availability Multiplier** — 23 redrob_signals. Availability acts as a MULTIPLIER (0.3–1.0x), not additive. Ghost candidate (inactive 180d) with perfect skills still scores only 43% max — they don't reach top 100.

Plus **Honeypot Detection**: timeline impossibility, keyword stuffer detection, impossible endorsement ratios.

## Scoring Formula

```
availability_mult = 0.30 + 0.70 × availability_score   # [0.3, 1.0]
raw_score = (0.45×skill + 0.35×career + 0.20×behavioral) × availability_mult
final_score = raw_score × honeypot_penalty
```

## Compute

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤5 min | ~2-3 min on 100K |
| Memory | ≤16 GB | ~1-2 GB peak |
| Compute | CPU only | ✅ Pure CPU |
| Network | Off | ✅ Zero API calls |

## Key Design Choices

- **No GPU, no LLM API**: Keyword expansion on career descriptions achieves semantic matching at 100x the speed of transformer inference on CPU.
- **Availability as multiplier not additive**: Prevents ghost candidates from floating to top on skill scores alone.
- **Career descriptions > skills section**: Skills are self-reported and gamed. Career descriptions reveal actual production experience.
- **Consulting firm penalty**: Full career at TCS/Infosys/Wipro = 0.1x company_type_score. Prior product-company experience restores it.
