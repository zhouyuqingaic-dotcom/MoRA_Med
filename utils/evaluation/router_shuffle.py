import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Any


KNOWN_LANGUAGE_KEYS = ("q_lang", "language", "lang")


def normalize_question(text: Any) -> str:
    """用于 shuffle 合法性检查；不改变真正送入 BioMedCLIP 的文本。"""
    text = unicodedata.normalize("NFKC", str(text))
    return " ".join(text.strip().casefold().split())


def _dataset_item(dataset, position: int) -> dict:
    item = dataset[position]
    if not isinstance(item, dict):
        raise TypeError(
            f"dataset[{position}] 应返回 dict，实际为 {type(item)}。"
        )
    return item


def detect_language_key(dataset) -> str:
    """
    只根据显式字段名检测语言，不根据问题字符猜测语言。
    """
    if len(dataset) == 0:
        raise ValueError("dataset 为空，无法检测语言字段。")

    observed = set()
    for position in range(min(len(dataset), 64)):
        raw_row = _dataset_item(dataset, position).get("raw_row", {})
        if isinstance(raw_row, dict):
            observed.update(str(key) for key in raw_row.keys())

    for key in KNOWN_LANGUAGE_KEYS:
        if key in observed:
            return key

    raise KeyError(
        "没有发现显式语言字段。已检查 q_lang / language / lang。"
        "请通过 language_key 指定真实字段；"
        "若当前 JSON 已明确为单语言集合，可传 language_key='none'。"
    )


def resolve_language_key(
    dataset,
    language_key: str | None = "auto",
) -> str | None:
    if language_key is None:
        return None

    value = str(language_key).strip()

    if not value or value.casefold() == "auto":
        return detect_language_key(dataset)

    if value.casefold() in {
        "none",
        "__all__",
        "single_language",
    }:
        return None

    return value


def _language_of(item: dict, language_key: str | None) -> str:
    if language_key is None:
        return "__all__"

    raw_row = item.get("raw_row", {})
    if not isinstance(raw_row, dict):
        raise TypeError("sample['raw_row'] 必须为 dict。")

    if language_key not in raw_row:
        raise KeyError(
            f"样本 index={item.get('index')} 缺少语言字段 {language_key!r}。"
        )

    value = str(raw_row[language_key]).strip().casefold()
    if not value:
        raise ValueError(
            f"样本 index={item.get('index')} 的语言字段为空。"
        )
    return value


def _collect_rows(dataset, language_key: str | None) -> list[dict]:
    rows = []
    seen = set()

    for position in range(len(dataset)):
        item = _dataset_item(dataset, position)
        index = int(item.get("index", position))
        if index in seen:
            raise ValueError(f"dataset 中出现重复 index={index}。")
        seen.add(index)

        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"样本 index={index} 的 question 为空。")

        rows.append(
            {
                "index": index,
                "question": question,
                "question_norm": normalize_question(question),
                "language": _language_of(item, language_key),
            }
        )

    return rows


def _derange_group(rows: list[dict], rng: random.Random) -> dict[int, int]:
    """
    为一个语言组构造一一对应的合法 permutation：
    - 不映射到自身；
    - 不映射到相同规范化问题；
    - 每个 target 只用一次。
    """
    if len(rows) < 2:
        language = rows[0]["language"] if rows else "UNKNOWN"
        raise ValueError(
            f"语言组 {language!r} 只有 {len(rows)} 个样本，无法做 derangement。"
        )

    candidates = {}
    for src_pos, source in enumerate(rows):
        valid = [
            tgt_pos
            for tgt_pos, target in enumerate(rows)
            if (
                target["index"] != source["index"]
                and target["question_norm"] != source["question_norm"]
            )
        ]
        if not valid:
            raise ValueError(
                "没有合法 shuffle target："
                f"index={source['index']}, language={source['language']!r}。"
            )
        rng.shuffle(valid)
        candidates[src_pos] = valid

    source_order = list(range(len(rows)))
    rng.shuffle(source_order)
    source_order.sort(key=lambda pos: len(candidates[pos]))

    target_to_source = {}

    def augment(source_pos: int, seen_targets: set[int]) -> bool:
        """用显式栈搜索增广路径，避免大型测试集触发递归深度限制。

        保持原有 candidates 顺序、seen_targets 语义及回溯时的匹配更新；
        不修改外部接口、随机种子或置换合法性约束。
        """
        # (当前 source, 下一个待检查候选的迭代器, 从父帧进入的 target)
        frames = [(source_pos, iter(candidates[source_pos]), None)]

        while frames:
            current_source, candidate_iter, _ = frames[-1]
            try:
                target_pos = next(candidate_iter)
            except StopIteration:
                frames.pop()
                continue

            if target_pos in seen_targets:
                continue
            seen_targets.add(target_pos)

            if target_pos not in target_to_source:
                # 末端找到空闲 target，按递归返回时相同的顺序重新分配匹配。
                target_to_source[target_pos] = current_source
                for depth in range(len(frames) - 1, 0, -1):
                    parent_source = frames[depth - 1][0]
                    incoming_target = frames[depth][2]
                    target_to_source[incoming_target] = parent_source
                return True

            matched_source = target_to_source[target_pos]
            frames.append(
                (
                    matched_source,
                    iter(candidates[matched_source]),
                    target_pos,
                )
            )

        return False

    for source_pos in source_order:
        if not augment(source_pos, set()):
            raise RuntimeError(
                "无法构造满足约束的一一 Router question permutation。"
            )

    return {
        source_pos: target_pos
        for target_pos, source_pos in target_to_source.items()
    }


def build_shuffle_map(
    dataset,
    seed: int,
    language_key: str | None = "auto",
    dataset_name: str = "SLAKE-test",
) -> dict:
    resolved_key = resolve_language_key(dataset, language_key)
    rows = _collect_rows(dataset, resolved_key)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["language"], []).append(row)

    rng = random.Random(int(seed))
    mapping = {}

    for language in sorted(groups):
        group = sorted(groups[language], key=lambda item: item["index"])
        assignment = _derange_group(group, rng)

        for src_pos, tgt_pos in sorted(assignment.items()):
            source = group[src_pos]
            target = group[tgt_pos]
            mapping[str(source["index"])] = {
                "source_index": source["index"],
                "source_question": source["question"],
                "target_index": target["index"],
                "target_question": target["question"],
                "language": language,
            }

    payload = {
        "format_version": 1,
        "dataset": str(dataset_name),
        "dataset_size": len(rows),
        "shuffle_seed": int(seed),
        "language_key": resolved_key if resolved_key is not None else "__all__",
        "mapping": mapping,
    }

    validate_shuffle_map(dataset, payload, language_key=resolved_key)
    return payload


def compute_shuffle_map_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_shuffle_map(payload: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    return compute_shuffle_map_sha256(payload)


def load_shuffle_map(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Router shuffle map 不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("mapping"), dict):
        raise TypeError("Router shuffle map 必须包含 dict 类型的 'mapping'。")

    return payload


def validate_shuffle_map(
    dataset,
    payload: dict,
    language_key: str | None = "auto",
) -> None:
    mapping = payload.get("mapping")
    if not isinstance(mapping, dict):
        raise TypeError("payload['mapping'] 必须为 dict。")

    if language_key == "auto" and payload.get("language_key") not in {None, "", "auto"}:
        stored_key = str(payload["language_key"])
        resolved_key = None if stored_key == "__all__" else stored_key
    else:
        resolved_key = resolve_language_key(dataset, language_key)

    rows = _collect_rows(dataset, resolved_key)
    by_index = {row["index"]: row for row in rows}

    expected = set(by_index)
    actual_sources = set()
    target_indices = []

    for source_key, entry in mapping.items():
        if not isinstance(entry, dict):
            raise TypeError(f"mapping[{source_key!r}] 必须为 dict。")

        source_index = int(source_key)
        if source_index not in by_index:
            raise KeyError(f"map 中 source index={source_index} 不属于当前 dataset。")

        if int(entry.get("source_index", source_index)) != source_index:
            raise ValueError(f"mapping key 与 source_index 不一致：{source_index}")

        target_index = int(entry["target_index"])
        if target_index not in by_index:
            raise KeyError(f"target index={target_index} 不属于当前 dataset。")
        if target_index == source_index:
            raise ValueError(f"发现 self mapping：index={source_index}")

        source = by_index[source_index]
        target = by_index[target_index]

        if source["question_norm"] == target["question_norm"]:
            raise ValueError(
                f"发现相同问题文本的无效 shuffle：{source_index}->{target_index}"
            )

        stored_source_question = str(
            entry.get("source_question", source["question"])
        )
        if normalize_question(stored_source_question) != source["question_norm"]:
            raise ValueError(
                f"source index={source_index} 的 source_question 与 dataset 不一致。"
            )

        stored_target_question = str(entry.get("target_question", ""))
        if normalize_question(stored_target_question) != target["question_norm"]:
            raise ValueError(
                f"source index={source_index} 的 target_question 与 "
                f"target index={target_index} 的真实问题不一致。"
            )

        if source["language"] != target["language"]:
            raise ValueError(
                f"发现跨语言 shuffle：{source_index}->{target_index}。"
            )

        actual_sources.add(source_index)
        target_indices.append(target_index)

    if actual_sources != expected:
        missing = sorted(expected - actual_sources)
        extra = sorted(actual_sources - expected)
        raise ValueError(
            f"shuffle map source 覆盖不完整：missing={missing[:10]}, extra={extra[:10]}"
        )

    if len(target_indices) != len(set(target_indices)):
        raise ValueError("shuffle map 不是一一对应：存在重复 target_index。")

    if set(target_indices) != expected:
        missing = sorted(expected - set(target_indices))
        raise ValueError(
            f"shuffle map 不是完整 permutation，缺少 target={missing[:10]}。"
        )

    declared_size = payload.get("dataset_size")
    if declared_size is not None and int(declared_size) != len(rows):
        raise ValueError(
            f"dataset_size 不一致：map={declared_size}, current={len(rows)}。"
        )


def get_shuffle_entry(payload: dict, source_index: int) -> dict:
    key = str(int(source_index))
    mapping = payload.get("mapping", {})
    if key not in mapping:
        raise KeyError(f"shuffle map 中找不到 source index={source_index}。")
    return mapping[key]