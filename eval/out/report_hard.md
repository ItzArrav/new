# CV classification evaluation (hard set)

- Models that answered (from the usage log): `openai/gpt-4o-mini` x40; 4 failed provider attempts were retried on the next provider
- Configured provider chain: `openai,anthropic,openrouter`
- Run: 2026-10-07 15:36 UTC · tokens 36,233 in / 12,687 out · cost $0.0130
- Normal CVs: 40 (synthetic, labelled by construction)
- Primary category: accuracy 98% · macro precision 98% · macro recall 97%
- Rejected after validation (counted as wrong): 0
- Average confidence the model gave its primary category: 98% when right, 90% when wrong

## Primary category, per category

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PSS | 100% | 100% | 100% | 4 |
| DCS | 100% | 100% | 100% | 4 |
| Front-End Developer | 100% | 100% | 100% | 4 |
| Back-End Developer | 100% | 100% | 100% | 4 |
| Web Automation | 100% | 100% | 100% | 4 |
| Mobile Automation | 100% | 100% | 100% | 4 |
| AI Engineer | 100% | 75% | 86% | 4 |
| Chatbot Application Tester | 80% | 100% | 89% | 4 |
| Other | 100% | 100% | 100% | 8 |

## All categories (primary + secondary), per category

A category counts if it is the primary or a secondary one, both in the label and in the answer.

Micro precision 100% · micro recall 96%

| Category | Precision | Recall | Support |
|---|---|---|---|
| PSS | 100% | 100% | 5 |
| DCS | 100% | 100% | 5 |
| Front-End Developer | 100% | 100% | 4 |
| Back-End Developer | 100% | 100% | 5 |
| Web Automation | 100% | 100% | 5 |
| Mobile Automation | 100% | 100% | 5 |
| AI Engineer | 100% | 67% | 6 |
| Chatbot Application Tester | 100% | 100% | 5 |
| Other | 100% | 100% | 8 |

Mixed CVs: the expected secondary category was returned for 5 of 8.

## Misclassified (primary)

- `hard_mixed_ai_engineer__chatbot_application_tester.txt`: expected AI Engineer, got Chatbot Application Tester
