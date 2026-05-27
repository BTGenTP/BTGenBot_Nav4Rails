## Analyse token count : Llama-3.1 vs Qwen2.5

**Tokenizers utilisés :** `NousResearch/Meta-Llama-3.1-8B-Instruct` et `Qwen/Qwen2.5-7B-Instruct`

### Distribution tokens total (input + output + chat template)

| Tokenizer | mean | p50 | p90 | p95 | p99 | max | >2048 | >4096 | >8192 |
|---|---|---|---|---|---|---|---|---|---|
| llama | 790.4 | 439 | 1595 | 2510 | 6508 | 8530 | 6.57% | 2.02% | 0.17% |
| qwen | 799.3 | 436 | 1599 | 2513 | 6667 | 9894 | 6.73% | 2.36% | 0.34% |

### Ratio compression Qwen/Llama (output XML)

**Médiane ratio :** 1.000 → Qwen produit en médiane 0.0% moins de tokens que Llama pour le même XML.

### Décision pour le SFT

- Avec `max_seq_length=4096` (config papier) : Llama tronque **12 exemples** (2.0%), Qwen **14** (2.4%).
- Recommandation : conserver `max_seq_length=4096` pour fidélité au papier, documenter le taux de troncature dans le rapport. Si VRAM le permet en QLoRA sur RTX 3090, tester 8192 en ablation.
