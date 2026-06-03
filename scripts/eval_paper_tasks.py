#!/usr/bin/env python3
"""
Évaluation sur les 9 tâches du papier BTGenBot (Izzo 2024) — Section 5.

Le papier évalue les modèles sur 9 tâches standardisées (pas sur le dataset d'entraînement).
Sources : external/BTGenBot/bt_generator/config/example_retrieving.yaml (descriptions + BTs ref)
          external/BTGenBot/bt_validator/xml/tree*.xml (BTs pour le validateur C++)

Métriques fidèles au papier :
  - Syntaxe : XML valid + BT structure valid  (substitut Groot2)
  - Sémantique : Node-F1 + TED + LLM-judge   (substitut validateur C++ ROS2)

Usage:
    python scripts/eval_paper_tasks.py                        # zero-shot v1_paper
    python scripts/eval_paper_tasks.py --prompt v3_vocab_injected
    python scripts/eval_paper_tasks.py --one-shot
    python scripts/eval_paper_tasks.py --all                  # tous les combos
"""

import argparse
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import yaml
import zss

ROOT = Path(__file__).parent.parent
YAML_CONFIG = ROOT / "external/BTGenBot/bt_generator/config/example_retrieving.yaml"
VALIDATOR_XML = ROOT / "external/BTGenBot/bt_validator/xml"
PROMPTS_FILE = ROOT / "dataset/prompts.json"
RESULTS_DIR = ROOT / "results"

MODEL = "claude-opus-4-8"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

META_TAGS = {"TreeNodesModel", "input_port", "output_port", "inout_port", "root", "BehaviorTree"}

# Mapping tâche → fichier XML validateur (pour référence d'exécution)
TASK_TO_VALIDATOR_XML = {
    "navigation":            "tree1.xml",
    "navigation_priority":   "tree2.xml",
    "navigation_fallback":   "tree3.xml",
    "navigation_arm":        "tree4.xml",
    "exploration":           "tree5.xml",
    "manipulator_exploration":"tree6.xml",
    "active_vision_picking": "tree7.xml",
    "material_processing":   "tree8.xml",
    "multi_station_assembly":"tree9.xml",
}

# Mapping tâche → clés YAML (description + BT de référence sémantique)
TASK_YAML_KEYS = {
    "navigation":             ("navigation_desc",             "navigation"),
    "navigation_priority":    ("navigation_priority_desc",    "navigation_priority"),
    "navigation_fallback":    ("navigation_fallback_desc",    "navigation_fallback"),
    "navigation_arm":         ("navigation_arm_activity_desc","navigation_arm_activity"),
    "exploration":            ("exploration_desc",            "exploration"),
    "manipulator_exploration":("manipulator_exploration_desc","manipulator_exploration"),
    "active_vision_picking":  ("active_vision_picking_desc",  "active_vision_picking"),
    "material_processing":    ("material_processing_desc",    "material_processing"),
    "multi_station_assembly": ("multi_station_assembly_desc", "multi_station_assembly"),
}

TASK_LABELS = {
    "navigation":             "T1 — Navigation",
    "navigation_priority":    "T2 — Navigation priorité",
    "navigation_fallback":    "T3 — Navigation fallback",
    "navigation_arm":         "T4 — Navigation + bras",
    "exploration":            "T5 — Exploration",
    "manipulator_exploration":"T6 — Exploration manipulateur",
    "active_vision_picking":  "T7 — Vision active + saisie",
    "material_processing":    "T8 — Traitement matériaux",
    "multi_station_assembly": "T9 — Assemblage multi-stations",
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_system_prompt(version: str) -> str:
    with open(PROMPTS_FILE) as f:
        return json.load(f)[version]["system"]


def call_model(system: str, user_input: str, effort: str = "xhigh") -> str:
    result = subprocess.run(
        ["claude", "-p", user_input,
         "--system-prompt", system,
         "--model", MODEL,
         "--effort", effort],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr[:200]}")
    return result.stdout


def call_judge(description: str, bt_xml: str) -> tuple[str, bool]:
    judge_system = (
        'Given a description of a behavior tree (BT) in natural language and a BT in XML format, '
        'say if the description matches the tree. Output only "Correct" or "Incorrect".'
    )
    user_msg = f"Description:\n{description}\n\nBehavior Tree XML:\n{bt_xml}"
    result = subprocess.run(
        ["claude", "-p", user_msg,
         "--system-prompt", judge_system,
         "--model", JUDGE_MODEL,
         "--effort", "low"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return "ERROR", False
    raw = result.stdout.strip()
    return raw, raw.lower().startswith("correct")

# ---------------------------------------------------------------------------
# XML parsing & metrics (identiques à baseline_claude_opus.py)
# ---------------------------------------------------------------------------

import re

def extract_xml(text: str) -> str:
    m = re.search(r"```(?:xml)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<root.*?</root>)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def is_xml_valid(text: str) -> bool:
    try:
        ET.fromstring(text)
        return True
    except ET.ParseError:
        return False


def is_bt_structure_valid(text: str) -> bool:
    """Structure minimale : <root><BehaviorTree> + SubTree refs résolues (Groot2-like).

    Résolution : BehaviorTree definitions + TreeNodesModel SubTree declarations.
    On cherche les SubTree refs DANS les BehaviorTree (pas dans TreeNodesModel lui-même).
    """
    try:
        root = ET.fromstring(text)
        if root.tag != "root" or root.find("BehaviorTree") is None:
            return False
        bt_ids = {bt.get("ID") for bt in root.findall("BehaviorTree")}
        tnm = root.find("TreeNodesModel")
        tnm_ids = {st.get("ID") for st in tnm.findall("SubTree")} if tnm is not None else set()
        used = set()
        for bt in root.findall("BehaviorTree"):
            used |= {st.get("ID") for st in bt.iter("SubTree") if st.get("ID")}
        unresolved = used - bt_ids - tnm_ids
        return len(unresolved) == 0
    except Exception:
        return False


def get_node_types(xml_str: str) -> Counter:
    try:
        root = ET.fromstring(xml_str)
        return Counter(el.tag for el in root.iter() if el.tag not in META_TAGS)
    except Exception:
        return Counter()


def compute_node_f1(pred_xml: str, ref_xml: str) -> float:
    pred = get_node_types(pred_xml)
    ref  = get_node_types(ref_xml)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = sum((pred & ref).values())
    precision = tp / sum(pred.values())
    recall    = tp / sum(ref.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _xml_to_zss(elem: ET.Element) -> zss.Node:
    node = zss.Node(elem.tag)
    for child in elem:
        if child.tag not in META_TAGS:
            node.addkid(_xml_to_zss(child))
    return node


def compute_ted(pred_xml: str, ref_xml: str) -> int:
    try:
        pred_root = ET.fromstring(pred_xml)
        ref_root  = ET.fromstring(ref_xml)
        pred_bt = pred_root.find("BehaviorTree") or pred_root
        ref_bt  = ref_root.find("BehaviorTree")  or ref_root
        return int(zss.simple_distance(_xml_to_zss(pred_bt), _xml_to_zss(ref_bt)))
    except Exception:
        return -1

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(prompt_version: str, one_shot: bool) -> dict:
    with open(YAML_CONFIG) as f:
        yaml_data = yaml.safe_load(f)

    system = load_system_prompt(prompt_version)
    mode = "one-shot" if one_shot else "zero-shot"
    results = []

    # Pour le one-shot, utiliser navigation comme exemple fixe (tâche la plus simple)
    one_shot_desc = yaml_data.get("navigation_desc", "")
    one_shot_bt   = yaml_data.get("navigation", "")

    print(f"\n=== Évaluation 9 tâches papier — {MODEL} | {prompt_version} | {mode} ===")

    for task_key, (desc_key, bt_key) in TASK_YAML_KEYS.items():
        description  = yaml_data.get(desc_key, "").strip()
        reference_bt = yaml_data.get(bt_key, "").strip()
        label        = TASK_LABELS[task_key]

        if not description:
            print(f"  SKIP {label}: description manquante dans YAML")
            continue

        # Construction du prompt utilisateur
        if one_shot and task_key != "navigation":
            user_msg = (
                f"Example description:\n{one_shot_desc}\n\n"
                f"Example XML:\n{one_shot_bt}\n\n"
                f"Now generate the XML for this description:\n{description}"
            )
        else:
            user_msg = description

        print(f"  [{label}] ... ", end="", flush=True)
        t0 = time.time()
        raw = call_model(system, user_msg)
        elapsed = time.time() - t0

        gen_xml = extract_xml(raw)

        # Métriques syntaxiques
        xml_ok  = is_xml_valid(gen_xml)
        bt_ok   = is_bt_structure_valid(gen_xml)

        # Métriques sémantiques vs référence YAML
        f1   = round(compute_node_f1(gen_xml, reference_bt), 4)
        ted  = compute_ted(gen_xml, reference_bt)

        # LLM-judge : le BT généré correspond-il à la description ?
        judge_raw, judge_correct = call_judge(description, gen_xml)

        verdict = "✓" if judge_correct else "✗"
        print(f"xml={xml_ok} bt={bt_ok} f1={f1:.3f} ted={ted} judge={verdict} ({elapsed:.0f}s)")

        # Référence validateur C++
        validator_xml_file = VALIDATOR_XML / TASK_TO_VALIDATOR_XML[task_key]
        validator_ref = validator_xml_file.read_text() if validator_xml_file.exists() else ""

        results.append({
            "task_key":          task_key,
            "task_label":        label,
            "description":       description,
            "reference_xml":     reference_bt,
            "validator_xml":     validator_ref,
            "generated_raw":     raw,
            "generated_xml":     gen_xml,
            "xml_valid":         xml_ok,
            "bt_structure_valid":bt_ok,
            "node_f1":           f1,
            "ted":               ted,
            "llm_judge_raw":     judge_raw,
            "llm_judge_correct": judge_correct,
            "elapsed_s":         round(elapsed, 1),
        })
        time.sleep(0.5)

    return {
        "model":          MODEL,
        "prompt_version": prompt_version,
        "mode":           mode,
        "run_date":       str(date.today()),
        "n_tasks":        len(results),
        "results":        results,
    }


def compute_and_print_summary(data: dict) -> dict:
    results = data["results"]
    n = len(results)
    if n == 0:
        return {}

    xml_ok   = [r["xml_valid"]          for r in results]
    bt_ok    = [r["bt_structure_valid"]  for r in results]
    f1s      = [r["node_f1"]             for r in results]
    teds     = [r["ted"]                 for r in results if r["ted"] >= 0]
    judges   = [r["llm_judge_correct"]   for r in results]

    summary = {
        "xml_valid_rate":            round(sum(xml_ok)  / n, 4),
        "bt_structure_valid_rate":   round(sum(bt_ok)   / n, 4),
        "node_f1_mean":              round(mean(f1s),   4),
        "node_f1_std":               round(stdev(f1s)   if len(f1s) > 1 else 0.0, 4),
        "ted_mean":                  round(mean(teds),  4) if teds else None,
        "ted_std":                   round(stdev(teds)  if len(teds) > 1 else 0.0, 4) if teds else None,
        "llm_judge_accuracy":        round(sum(judges)  / n, 4),
    }
    data["summary"] = summary
    return summary


def print_comparison_table(all_data: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("TABLEAU COMPARATIF — 9 tâches papier BTGenBot")
    print("Papier syntaxe = Groot2 (Table 4) | sémantique = experts humains (Table 5, Section 5.3)")
    print("⚠  Colonne Sémantique NON COMPARABLE : nous=Opus LLM-judge ; papier=experts humains")
    print("=" * 90)
    hdr = f"{'Config':<32} {'XML':>6} {'BTstruct':>9} {'NodeF1':>8} {'TED':>7} {'Semantique':>11}"
    print(hdr)
    print("-" * 90)

    for d in all_data:
        s = d.get("summary", {})
        if not s:
            continue
        label   = f"{d['prompt_version']} ({d['mode']})"
        xml_r   = f"{s['xml_valid_rate']*100:.0f}%"
        bts_r   = f"{s['bt_structure_valid_rate']*100:.0f}%"
        f1      = f"{s['node_f1_mean']:.3f}±{s['node_f1_std']:.3f}"
        ted     = f"{s['ted_mean']:.1f}" if s.get("ted_mean") is not None else "N/A"
        judge   = f"{s['llm_judge_accuracy']*100:.1f}% (Opus)"
        print(f"{label:<32} {xml_r:>6} {bts_r:>9} {f1:>12} {ted:>7} {judge:>11}")

    print("-" * 90)
    print(f"{'ChatGPT ZS (papier T4&T5)':<32} {'100%':>6} {'—':>9} {'—':>12} {'—':>7} {'77.8% (hum)':>11}")
    print(f"{'Gemini ZS (papier T4&T5)':<32} {'88.9%':>6} {'—':>9} {'—':>12} {'—':>7} {'55.6% (hum)':>11}")
    print(f"{'LLaMA-2-13b ZS (papier T4&T5)':<32} {'33.3%':>6} {'—':>9} {'—':>12} {'—':>7} {'22.2% (hum)':>11}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Évaluation 9 tâches papier BTGenBot.")
    parser.add_argument("--prompt", default="v1_paper",
                        choices=["v1_paper", "v2_btcpp_enriched", "v3_vocab_injected", "v4_action_aware"])
    parser.add_argument("--one-shot", dest="one_shot", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="Lance tous les combos prompt×mode.")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Ne relance pas l'inférence, charge depuis results/ et affiche tableau.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.all:
        combos = [
            ("v1_paper",          False),
            ("v4_action_aware",   False),
            ("v3_vocab_injected", False),
            ("v1_paper",          True),
            ("v4_action_aware",   True),
        ]
    else:
        combos = [(args.prompt, args.one_shot)]

    all_data = []
    for prompt_version, one_shot in combos:
        mode = "one-shot" if one_shot else "zero-shot"
        out_file = RESULTS_DIR / f"paper_tasks_{prompt_version}_{mode}.json"

        if args.skip_inference and out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
            if "summary" not in data:
                compute_and_print_summary(data)
        else:
            data = run_evaluation(prompt_version, one_shot)
            compute_and_print_summary(data)
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  → Sauvegardé : {out_file}")

        all_data.append(data)

    print_comparison_table(all_data)


if __name__ == "__main__":
    main()
