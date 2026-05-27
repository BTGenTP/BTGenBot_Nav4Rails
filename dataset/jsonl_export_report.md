## Export JSONL trainable

**Dataset :** 594 exemples (fidélité papier BTGenBot, aucun filtrage appliqué).

### Fichiers générés

| Fichier | Format | Lignes | Taille |
|---|---|---|---|
| `dataset/btgenbot_chat.jsonl` | chat (messages) | 594 | 2360 KB |
| `dataset/btgenbot_alpaca.jsonl` | alpaca | 594 | 2317 KB |

### Format chat (utilisé pour SFT)

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
], "metadata": {...}}
```

### System prompt (v1 papier)

```
You will be provided a summary of a task performed by a behavior tree, and your objective is to express this behavior tree in XML format.
```

### Métadonnées embarquées

`record_id, category, has_recovery_pattern, in_smoke_test, xml_valid_strict, bt_node_count, bt_max_depth, tok_llama_total, tok_qwen_total`

Le SFTTrainer ignore ces champs (lit juste `messages` ou `instruction/input/output`),
mais ils permettent du filtrage et de l'analyse downstream sans réjoindre les CSV.

### Usage SFT (exemple TRL)

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="btgenbot_chat.jsonl", split="train")
# SFTTrainer lira le champ messages et appliquera le chat_template du tokenizer
```
