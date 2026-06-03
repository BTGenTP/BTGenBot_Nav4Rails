#!/usr/bin/env python3
"""
Évaluation des modèles fine-tunés du papier BTGenBot (Izzo 2024) sur même référentiel.

Extrait les BTs générés depuis les 4 PDFs du repo :
  external/BTGenBot/prompt/{zero,one}_shot_{llamachat,codellama}_results.pdf

Calcule Node-F1 + TED + LLM-judge (Opus 4.8) avec le même code que eval_paper_tasks.py,
référence = example_retrieving.yaml (identique à notre benchmark).

Usage:
    python scripts/eval_paper_models.py                # extraction + métriques, pas de judge
    python scripts/eval_paper_models.py --judge        # + LLM-judge Opus 4.8
    python scripts/eval_paper_models.py --skip-extract # charge depuis results/ existants
"""

import argparse
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import pdfplumber
import yaml
import zss

ROOT = Path(__file__).parent.parent
YAML_CONFIG = ROOT / "external/BTGenBot/bt_generator/config/example_retrieving.yaml"
PROMPTS_DIR = ROOT / "external/BTGenBot/prompt"
RESULTS_DIR = ROOT / "results"

JUDGE_MODEL = "claude-opus-4-8"

META_TAGS = {"TreeNodesModel", "input_port", "output_port", "inout_port", "root", "BehaviorTree", "SubTree"}

TASK_YAML_KEYS = {
    1: ("navigation_desc",             "navigation"),
    2: ("navigation_priority_desc",    "navigation_priority"),
    3: ("navigation_fallback_desc",    "navigation_fallback"),
    4: ("navigation_arm_activity_desc","navigation_arm_activity"),
    5: ("exploration_desc",            "exploration"),
    6: ("manipulator_exploration_desc","manipulator_exploration"),
    7: ("active_vision_picking_desc",  "active_vision_picking"),
    8: ("material_processing_desc",    "material_processing"),
    9: ("multi_station_assembly_desc", "multi_station_assembly"),
}

CONFIGS = [
    ("zero_shot_llamachat_results.pdf",  "llamachat_ft", "zero-shot"),
    ("one_shot_llamachat_results.pdf",   "llamachat_ft", "one-shot"),
    ("zero_shot_codellama_results.pdf",  "codellama_ft", "zero-shot"),
    ("one_shot_codellama_results.pdf",   "codellama_ft", "one-shot"),
]

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_bts(pdf_path: Path) -> dict[int, str]:
    """Retourne {task_num: raw_output_text} pour les 9 tâches."""
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    segments = re.split(r'\nTask (\d+)\n', full_text)
    results = {}
    for i in range(1, len(segments) - 1, 2):
        task_num = int(segments[i])
        body = segments[i + 1]
        # Le modèle répète le prompt complet avant sa réponse.
        # La réponse réelle est après le dernier [/INST].
        parts = body.split("[/INST]")
        raw_output = parts[-1] if len(parts) > 1 else body
        results[task_num] = raw_output.strip()
    return results


# ---------------------------------------------------------------------------
# XML parsing (identique à eval_paper_tasks.py)
# ---------------------------------------------------------------------------

def extract_xml(text: str) -> str:
    m = re.search(r"```(?:xml)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<root.*?</root>)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: chercher <root ... > ... </root> même avec tag malformé
    m = re.search(r"(<root[^>]*>.*)", text, re.DOTALL)
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
        # SubTree refs dans les BehaviorTree seulement (pas dans TreeNodesModel)
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
# LLM-judge
# ---------------------------------------------------------------------------

def call_judge(description: str, bt_xml: str) -> tuple[str, bool]:
    judge_system = (
        "Given a description of a behavior tree (BT) in natural language and a BT in XML format, "
        "say if the description matches the tree. Output only \"Correct\" or \"Incorrect\"."
    )
    user_msg = f"Description:\n{description}\n\nBehavior Tree XML:\n{bt_xml}"
    result = subprocess.run(
        ["claude", "-p", user_msg,
         "--system-prompt", judge_system,
         "--model", JUDGE_MODEL,
         "--effort", "high"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return "ERROR", False
    raw = result.stdout.strip()
    return raw, raw.lower().startswith("correct")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(pdf_filename: str, model_name: str, mode: str,
                    yaml_data: dict, run_judge: bool) -> dict:
    pdf_path = PROMPTS_DIR / pdf_filename
    print(f"\n=== {model_name} | {mode} | {pdf_filename} ===")

    raw_outputs = extract_pdf_bts(pdf_path)
    results = []

    for task_num, (desc_key, bt_key) in TASK_YAML_KEYS.items():
        description  = yaml_data.get(desc_key, "").strip()
        reference_bt = yaml_data.get(bt_key, "").strip()

        raw_output = raw_outputs.get(task_num, "")
        gen_xml    = extract_xml(raw_output)

        xml_ok = is_xml_valid(gen_xml)
        bt_ok  = is_bt_structure_valid(gen_xml)
        f1     = round(compute_node_f1(gen_xml, reference_bt), 4)
        ted    = compute_ted(gen_xml, reference_bt)

        judge_raw, judge_correct = ("—", None)
        if run_judge and xml_ok:
            judge_raw, judge_correct = call_judge(description, gen_xml)
            time.sleep(0.3)

        verdict = "✓" if judge_correct else ("✗" if judge_correct is False else "—")
        print(f"  T{task_num}: xml={xml_ok} bt={bt_ok} f1={f1:.3f} ted={ted:>3} judge={verdict}")

        results.append({
            "task_num":          task_num,
            "description":       description,
            "reference_xml":     reference_bt,
            "raw_output":        raw_output,
            "generated_xml":     gen_xml,
            "xml_valid":         xml_ok,
            "bt_structure_valid":bt_ok,
            "node_f1":           f1,
            "ted":               ted,
            "llm_judge_raw":     judge_raw,
            "llm_judge_correct": judge_correct,
        })

    return {
        "model":      model_name,
        "mode":       mode,
        "source_pdf": pdf_filename,
        "run_date":   str(date.today()),
        "n_tasks":    len(results),
        "results":    results,
    }


def compute_summary(data: dict) -> dict:
    results = data["results"]
    n = len(results)
    if n == 0:
        return {}

    xml_ok  = [r["xml_valid"]          for r in results]
    bt_ok   = [r["bt_structure_valid"]  for r in results]
    f1s     = [r["node_f1"]             for r in results]
    teds    = [r["ted"]                 for r in results if r["ted"] >= 0]
    judges  = [r["llm_judge_correct"]   for r in results if r["llm_judge_correct"] is not None]

    summary = {
        "xml_valid_rate":          round(sum(xml_ok) / n, 4),
        "bt_structure_valid_rate": round(sum(bt_ok)  / n, 4),
        "node_f1_mean":            round(mean(f1s),  4),
        "node_f1_std":             round(stdev(f1s)  if len(f1s) > 1 else 0.0, 4),
        "ted_mean":                round(mean(teds), 4) if teds else None,
        "ted_std":                 round(stdev(teds) if len(teds) > 1 else 0.0, 4) if teds else None,
        "llm_judge_accuracy":      round(sum(judges) / len(judges), 4) if judges else None,
    }
    data["summary"] = summary
    return summary


def print_table(paper_data: list[dict], our_data: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF — même référentiel (example_retrieving.yaml + Opus 4.8 judge)")
    print("Syntaxe : XML parse + BT structure | Sémantique : Node-F1, TED, LLM-judge Opus 4.8")
    print("=" * 100)
    hdr = f"{'Config':<36} {'XML':>6} {'BTstruct':>9} {'NodeF1':>13} {'TED':>7} {'Judge':>8}"
    print(hdr)
    print("-" * 100)

    def fmt_row(label, s):
        xml_r  = f"{s['xml_valid_rate']*100:.0f}%"
        bts_r  = f"{s['bt_structure_valid_rate']*100:.0f}%"
        f1     = f"{s['node_f1_mean']:.3f}±{s['node_f1_std']:.3f}"
        ted    = f"{s['ted_mean']:.1f}" if s.get("ted_mean") is not None else "N/A"
        judge  = f"{s['llm_judge_accuracy']*100:.1f}%" if s.get("llm_judge_accuracy") is not None else "—"
        print(f"{label:<36} {xml_r:>6} {bts_r:>9} {f1:>13} {ted:>7} {judge:>8}")

    print("── Modèles fine-tunés du papier (nos métriques) ──")
    for d in paper_data:
        s = d.get("summary", {})
        if s:
            fmt_row(f"{d['model']} ({d['mode']})", s)

    print("── Notre baseline (Opus 4.8) ──")
    for d in our_data:
        s = d.get("summary", {})
        if s:
            fmt_row(f"opus4.8_{d['prompt_version']} ({d['mode']})", s)

    print("── Papier (Table 4&5, experts humains) ──")
    print(f"{'chatgpt_zs (Table4: syntaxe Groot2)':<36} {'100%':>6} {'—':>9} {'—':>13} {'—':>7} {'77.8%†':>8}")
    print(f"{'gemini_zs':<36} {'88.9%':>6} {'—':>9} {'—':>13} {'—':>7} {'55.6%†':>8}")
    print(f"{'llama2-13b_zs':<36} {'33.3%':>6} {'—':>9} {'—':>13} {'—':>7} {'22.2%†':>8}")
    print("=" * 100)
    print("† = experts humains (Table 5), pas Opus judge — non comparable directement")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge",        action="store_true", help="Lance LLM-judge Opus 4.8")
    parser.add_argument("--skip-extract", action="store_true", help="Charge depuis results/ existants")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    with open(YAML_CONFIG) as f:
        yaml_data = yaml.safe_load(f)

    paper_data = []
    for pdf_filename, model_name, mode in CONFIGS:
        out_file = RESULTS_DIR / f"paper_models_{model_name}_{mode.replace('-', '_')}.json"

        if args.skip_extract and out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
            if "summary" not in data:
                compute_summary(data)
        else:
            data = evaluate_config(pdf_filename, model_name, mode, yaml_data, args.judge)
            compute_summary(data)
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  → {out_file}")

        paper_data.append(data)

    # Charger nos résultats baseline pour comparatif
    our_data = []
    for fname in sorted(RESULTS_DIR.glob("paper_tasks_*.json")):
        with open(fname) as f:
            d = json.load(f)
        if "summary" in d:
            our_data.append(d)

    print_table(paper_data, our_data)


if __name__ == "__main__":
    main()
