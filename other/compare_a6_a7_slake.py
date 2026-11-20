#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLAKE A6 (dynamic RMS) vs A7 (train-calibrated fixed stream scales).

Read-only for original checkpoints and evaluation outputs. No GPU/API required.
Run from the MoRA_Med project root:
    python other/compare_a6_a7_slake.py

Optional paths:
    python other/compare_a6_a7_slake.py --model-root /path/to/Models/MoRA_Med \
        --eval-root /path/to/Models/MoRA_Med/Eval_SLAKE
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

METRICS = ("closed_acc", "open_strict_acc", "overall_strict_acc")
HEADERS = ("Closed", "Open", "Overall")
EXPECTED_A6 = (
    "_A6-DLR_",
    "_Scale-learned_",
    "_Gate-learned-Init-0.5_",
    "_Lambda-learnable-Init-0.9-Max-1_",
    "_RMS-1_",
    "_From-Stage1-Seed-2048_Seed-2048",
)
EXPECTED_A7 = (
    "_A7-DLR-StreamFixed-2p11582434-10-10-4p59969044_",
    "_From-Stage1-Seed-2048_Seed-2048",
)


def load_json(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover(root, eval_root):
    names = set()
    for folder in (root, eval_root):
        if folder.is_dir():
            for p in folder.glob("Stage2_SLAKE_*"):
                if p.is_dir():
                    names.add(p.name)
    return sorted(names)


def describe_run(root, name):
    folder = root / name
    checkpoint_count = len(list(folder.glob("checkpoint-*")))
    final = folder / "final_weights"
    has_final = (
        (final / "adapter_model.safetensors").is_file()
        and (final / "visual_adapter.pt").is_file()
    )
    return f"{name} [ckpts={checkpoint_count}, final_weights={'YES' if has_final else 'NO'}]"


def show_inventory(model_root, eval_root, names):
    print("\n=== A6/A7 Stage2 目录清单（只读） ===")
    for label in ("_A6", "_A7"):
        found = [n for n in names if label in n]
        print(f"\n{label[1:]}: {len(found)} 个")
        for n in found[:25]:
            print("  " + describe_run(model_root, n))
            eval_dir = eval_root / n
            print(
                "    eval: validation="
                + str((eval_dir / "validation" / "leaderboard.json").is_file())
                + ", test="
                + str((eval_dir / "test" / "leaderboard.json").is_file())
            )
        if len(found) > 25:
            print(f"  ... 还有 {len(found) - 25} 个")

    stage1 = sorted(
        p.name for p in model_root.glob("Stage1_MIMIC_CXR_*A6*")
        if p.is_dir()
    )
    print(f"\nA6 Stage1 目录: {len(stage1)} 个")
    for n in stage1[:25]:
        is_matched = (
            "_A6-DLR_" in n
            and "_Lambda-learnable-Init-0.9-Max-1_" in n
            and "_RMS-1_" in n
            and n.endswith("_Seed-2048")
        )
        print(f"  {'[可能匹配]' if is_matched else '[需检查]'} {n}")


def choose_run(names, requested, category, a7_prefix=None):
    if requested:
        name = Path(requested).name
        if name not in names:
            raise ValueError(f"指定的 {category} 目录不存在: {requested}")
        return name

    if category == "A7":
        matches = [n for n in names if all(tag in n for tag in EXPECTED_A7)]
    else:
        matches = [
            n for n in names
            if all(tag in n for tag in EXPECTED_A6)
            and (a7_prefix is None or n.startswith(a7_prefix + "_"))
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None
    raise ValueError(
        f"找到 {len(matches)} 个符合命名条件的 {category}，"
        f"请使用 --{category.lower()}-run 显式指定目录名。"
    )


def read_evaluation(eval_root, run_name):
    base = eval_root / run_name
    validation = load_json(base / "validation" / "leaderboard.json")
    test = load_json(base / "test" / "leaderboard.json")
    best = load_json(base / "validation" / "best_checkpoint.json")
    problems = []

    if not isinstance(validation, list) or not validation:
        problems.append("Validation leaderboard.json 缺失或为空")
    if not isinstance(test, list) or len(test) != 1:
        problems.append("Test leaderboard.json 必须恰好包含 1 个 checkpoint")
    if not isinstance(best, dict):
        problems.append("缺少 Validation best_checkpoint.json")

    best_entry = None
    test_entry = test[0] if isinstance(test, list) and len(test) == 1 else None

    if isinstance(validation, list) and validation:
        best_entry = max(validation, key=lambda e: e["overall_strict_acc"])
        if isinstance(best, dict):
            if best.get("selection_split") != "validation":
                problems.append("checkpoint 选择未标明来自 validation")
            if best.get("selected_checkpoint") != best_entry.get("checkpoint"):
                problems.append("best_checkpoint.json 与 validation 排行不一致")
            if best.get("ablation_id") not in (None, "A6" if "_A6" in run_name else "A7"):
                problems.append("best_checkpoint.json 中的 ablation_id 不匹配")
            if test_entry is not None and test_entry.get("checkpoint") != best.get("selected_checkpoint"):
                problems.append("Test 不是 Validation 选出的最佳 checkpoint")

    samples = None
    if test_entry is not None:
        p = base / "test" / str(test_entry["checkpoint"]) / "samples.jsonl"
        if p.is_file():
            with p.open("r", encoding="utf-8") as handle:
                samples = [json.loads(line) for line in handle if line.strip()]

    return {
        "name": run_name,
        "validation": best_entry,
        "test": test_entry,
        "selected": best,
        "samples": samples,
        "problems": problems,
    }


def show_scores(label, result):
    print(f"\n=== {label} ===")
    if result["selected"]:
        b = result["selected"]
        print("Validation 选择:", b.get("selected_checkpoint"),
              "| selection_metric:", b.get("selection_metric"))
    for phase in ("validation", "test"):
        row = result[phase]
        if row is None:
            print(f"{phase}: 尚未生成完整结果")
        else:
            parts = " | ".join(
                f"{label_}: {100 * float(row[key]):.2f}%"
                for key, label_ in zip(METRICS, HEADERS)
            )
            print(f"{phase}: {row['checkpoint']} | {parts}")
    if result["problems"]:
        for p in result["problems"]:
            print("⚠️  " + p)


def unique_sample_map(samples):
    result = {}
    for row in samples:
        key = (
            str(row.get("index")),
            str(row.get("image_path")),
            str(row.get("question")),
            str(row.get("gt_norm")),
            str(row.get("question_category")),
        )
        if key in result:
            raise ValueError("逐样本结果中出现重复样本键: " + repr(key))
        if "final_correct" not in row:
            raise ValueError("逐样本结果缺少 final_correct 字段")
        result[key] = row
    return result


def paired_analysis(a6, a7, output_dir):
    if a6["samples"] is None or a7["samples"] is None:
        return {"available": False, "reason": "一个或两个样本文件 samples.jsonl 缺失"}

    left = unique_sample_map(a6["samples"])
    right = unique_sample_map(a7["samples"])
    if set(left) != set(right):
        missing_a6 = len(set(right) - set(left))
        missing_a7 = len(set(left) - set(right))
        return {
            "available": False,
            "reason": (f"两次 Test 样本不完全相同：A6 缺 {missing_a6}，"
                       f"A7 缺 {missing_a7}。禁止据此做严格配对统计。"),
        }

    keys = sorted(left)
    count = len(keys)
    if count == 0:
        return {"available": False, "reason": "没有测试样本"}

    a6_only = a7_only = both_correct = both_wrong = 0
    changes = []
    differences = []
    for k in keys:
        x, y = left[k], right[k]
        x_ok, y_ok = bool(x["final_correct"]), bool(y["final_correct"])
        differences.append(int(y_ok) - int(x_ok))
        if x_ok and y_ok:
            both_correct += 1
        elif not x_ok and not y_ok:
            both_wrong += 1
        elif x_ok:
            a6_only += 1
        else:
            a7_only += 1
        if x_ok != y_ok:
            changes.append({
                "index": x.get("index"),
                "question_category": x.get("question_category"),
                "question": x.get("question"),
                "ground_truth": x.get("gt_raw"),
                "A6_prediction": x.get("pred_raw"),
                "A7_prediction": y.get("pred_raw"),
                "A6_correct": x_ok,
                "A7_correct": y_ok,
            })

    disagreements_path = output_dir / "a6_a7_disagreements.csv"
    with disagreements_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "question_category", "question", "ground_truth",
            "A6_prediction", "A7_prediction", "A6_correct", "A7_correct",
        ])
        writer.writeheader()
        writer.writerows(changes)

    # Paired nonparametric bootstrap CI: descriptive uncertainty for one seed.
    rng = random.Random(2048)
    bootstrap = []
    for _ in range(2000):
        bootstrap.append(
            100 * sum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    bootstrap.sort()
    return {
        "available": True,
        "test_samples": count,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "A6_only_correct": a6_only,
        "A7_only_correct": a7_only,
        "A7_minus_A6_pp": 100 * sum(differences) / count,
        "bootstrap_95pct_CI_pp": [
            bootstrap[int(0.025 * (len(bootstrap) - 1))],
            bootstrap[int(0.975 * (len(bootstrap) - 1))],
        ],
        "disagreements_csv": str(disagreements_path),
    }


def main():
    p = argparse.ArgumentParser(description="SLAKE A6/A7 read-only evaluation comparison")
    p.add_argument("--model-root", type=Path,
                   default=Path("/home/yuqing/Models/MoRA_Med"))
    p.add_argument("--eval-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("logs/a6_a7_comparison"))
    p.add_argument("--a6-run", type=str, default=None,
                   help="Stage2 run directory NAME (or full path)")
    p.add_argument("--a7-run", type=str, default=None,
                   help="Stage2 run directory NAME (or full path)")
    args = p.parse_args()
    model_root = args.model_root.expanduser()
    eval_root = (args.eval_root or (model_root / "Eval_SLAKE")).expanduser()
    output_dir = args.output_dir.expanduser()
    names = discover(model_root, eval_root)
    show_inventory(model_root, eval_root, names)

    a7_run = choose_run(names, args.a7_run, "A7")
    if a7_run is None:
        print("\n❌ 未找到 A7 seed2048 固定系数实验目录；请检查 --model-root。")
        return 2

    a7_prefix = a7_run.split("_A7-DLR-StreamFixed-", 1)[0]
    a6_run = choose_run(names, args.a6_run, "A6", a7_prefix=a7_prefix)
    if a6_run is None:
        print("\n⚠️  尚未发现与 A7 在目录命名上匹配的 A6:")
        print("    必须为 A6-DLR + dynamic RMS + lambda learnable init 0.9")
        print("    + 相同 160k MIMIC 缓存 + Stage1/Stage2 seed 2048。")
        print("    历史 Lambda-fixed-1 的 A6 不属于严格对照，不能只改文件名。")
        print("    如 A6 Stage1/Stage2 权重不存在，需要训练匹配的 A6。")
        print("    有相应权重但尚无评测时，只需跑它的 Validation -> Test。")
        return 0

    print("\n=== 自动匹配到的实验 ===")
    print("A6:", a6_run)
    print("A7:", a7_run)
    print("\n注意：目录命名只能初筛；最终仍须核对 A6/A7 的训练日志、")
    print("两阶段有效 batch、训练步数、代码版本和 LLM Judge 设置。")

    a6 = read_evaluation(eval_root, a6_run)
    a7 = read_evaluation(eval_root, a7_run)
    show_scores("A6 (dynamic RMS)", a6)
    show_scores("A7 (fixed stream scales)", a7)

    if a6["problems"] or a7["problems"] or a6["test"] is None or a7["test"] is None:
        print("\n⚠️  评测文件不齐全或选模验证未通过，暂不生成正式对照结果。")
        print("    如 A6 权重已经存在，请先运行对应的 A6 Validation 和 Test。")
        return 0

    rows = []
    for phase in ("validation", "test"):
        for metric, label in zip(METRICS, HEADERS):
            s6 = float(a6[phase][metric])
            s7 = float(a7[phase][metric])
            rows.append({
                "split": phase,
                "metric": label,
                "A6_accuracy_pct": round(100 * s6, 4),
                "A7_accuracy_pct": round(100 * s7, 4),
                "A7_minus_A6_pp": round(100 * (s7 - s6), 4),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "a6_a7_accuracy_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    paired = paired_analysis(a6, a7, output_dir)
    report = {
        "A6_run": a6_run,
        "A7_run": a7_run,
        "A6_selected_checkpoint": a6["selected"]["selected_checkpoint"],
        "A7_selected_checkpoint": a7["selected"]["selected_checkpoint"],
        "accuracy_comparison": rows,
        "paired_test": paired,
        "note": ("Matching names alone do not establish matched training conditions. "
                 "Single-seed results do not establish equivalence. "
                 "Re-run LLM judging under the same model/prompt/settings if needed."),
    }
    report_path = output_dir / "a6_a7_comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== A7 - A6（百分点 pp） ===")
    for row in rows:
        print(
            f"{row['split']:10s} {row['metric']:7s}: "
            f"A6={row['A6_accuracy_pct']:.2f}%  "
            f"A7={row['A7_accuracy_pct']:.2f}%  "
            f"差={row['A7_minus_A6_pp']:+.2f} pp"
        )
    if paired.get("available"):
        print("\n逐样本配对统计 (Test):")
        print("  样本数:", paired["test_samples"])
        print("  仅 A6 正确:", paired["A6_only_correct"])
        print("  仅 A7 正确:", paired["A7_only_correct"])
        print("  A7 - A6:", f"{paired['A7_minus_A6_pp']:+.2f} pp")
        print("  配对 bootstrap 95% CI:",
              [round(v, 3) for v in paired["bootstrap_95pct_CI_pp"]], "pp")
    else:
        print("逐样本配对统计不可用:", paired.get("reason"))
    print("\n已保存：", csv_path, "和", report_path, sep="\n  ")
    print("\n提醒：单个 seed 的分数接近，不等于已证明两方案统计等效。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print("\nERROR:", e, file=sys.stderr)
        sys.exit(2)

"""
/home/yuqing/miniconda3/envs/qwen3vl/bin/python /home/yuqing/RemoteProjects/MoRA_Med/other/compare_a6_a7_slake.py 

=== A6/A7 Stage2 目录清单（只读） ===

A6: 8 个
  Stage2_SLAKE_mimic_cxr_train_clean_v2_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=0, final_weights=NO]
    eval: validation=False, test=False
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_From-Stage1-Seed-1024_Seed-1024 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_From-Stage1-Seed-4096_Seed-4096 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-1-Max-1.5_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=0, final_weights=NO]
    eval: validation=False, test=False
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen80k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_From-Stage1-Seed-2048_Seed-2048 [ckpts=0, final_weights=NO]
    eval: validation=False, test=False

A7: 1 个
  Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A7-DLR-StreamFixed-2p11582434-10-10-4p59969044_From-Stage1-Seed-2048_Seed-2048 [ckpts=4, final_weights=YES]
    eval: validation=True, test=True

A6 Stage1 目录: 8 个
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_Seed-2048
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_Seed-2048
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_Seed-1024
  [可能匹配] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_Seed-2048
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_Seed-4096
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-1-Max-1.5_RMS-1_Seed-2048
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen80k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_Seed-2048
  [需检查] Stage1_MIMIC_CXR_mimic_cxr_train_clean_v2_screen80k_seed2048_A6_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-fixed-1_RMS-1_Seed-2048_UnifiedLearnedRate

=== 自动匹配到的实验 ===
A6: Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A6-DLR_Experts-Conv2D-F3_F5_F7_Scale-learned_Gate-learned-Init-0.5_Lambda-learnable-Init-0.9-Max-1_RMS-1_From-Stage1-Seed-2048_Seed-2048
A7: Stage2_SLAKE_mimic_cxr_train_clean_v2_screen160k_seed2048_A7-DLR-StreamFixed-2p11582434-10-10-4p59969044_From-Stage1-Seed-2048_Seed-2048

注意：目录命名只能初筛；最终仍须核对 A6/A7 的训练日志、
两阶段有效 batch、训练步数、代码版本和 LLM Judge 设置。

=== A6 (dynamic RMS) ===
Validation 选择: checkpoint-1845 | selection_metric: overall_strict_acc
validation: checkpoint-1845 | Closed: 92.18% | Open: 87.89% | Overall: 89.61%
test: checkpoint-1845 | Closed: 91.03% | Open: 89.51% | Overall: 90.11%

=== A7 (fixed stream scales) ===
Validation 选择: checkpoint-1845 | selection_metric: overall_strict_acc
validation: checkpoint-1845 | Closed: 91.82% | Open: 89.08% | Overall: 90.19%
test: checkpoint-1845 | Closed: 90.91% | Open: 89.11% | Overall: 89.83%

=== A7 - A6（百分点 pp） ===
validation Closed : A6=92.18%  A7=91.82%  差=-0.36 pp
validation Open   : A6=87.89%  A7=89.08%  差=+1.20 pp
validation Overall: A6=89.61%  A7=90.19%  差=+0.57 pp
test       Closed : A6=91.03%  A7=90.91%  差=-0.12 pp
test       Open   : A6=89.51%  A7=89.11%  差=-0.40 pp
test       Overall: A6=90.11%  A7=89.83%  差=-0.29 pp

逐样本配对统计 (Test):
  样本数: 2094
  仅 A6 正确: 49
  仅 A7 正确: 43
  A7 - A6: -0.29 pp
  配对 bootstrap 95% CI: [-1.194, 0.573] pp

已保存：
  logs/a6_a7_comparison/a6_a7_accuracy_comparison.csv
  和
  logs/a6_a7_comparison/a6_a7_comparison.json

提醒：单个 seed 的分数接近，不等于已证明两方案统计等效。

Process finished with exit code 0

"""