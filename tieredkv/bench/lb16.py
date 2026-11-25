# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""lb16: the 16 English LongBench tasks for tb_bench (drop-in for lb.py: PROMPT, MAXGEN, DATA, score).

Prompts and generation lengths are read from the official THUDM/LongBench config files (commit 2e00731f, MIT) shipped
next to this file (dataset2prompt.json, dataset2maxlen.json). The metric functions are the English ones of the official
metrics.py, copied without change of logic (the Chinese ones, which need jieba, are left out), and score() applies the
official eval.py scorer: first line only for trec / triviaqa / samsum (lsht is Chinese and not used), maximum over the
reference answers. NO_CHAT lists the tasks the official pred.py feeds WITHOUT a chat template.
Needs `rouge` and `fuzzywuzzy` (pure Python); if LB16_PYLIB is set it is put on sys.path first (tb_bench runs with -I).
Run `python lb16.py selftest` for the metric checks.
"""
from __future__ import annotations

import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

if os.environ.get("LB16_PYLIB"):
    sys.path.insert(0, os.environ["LB16_PYLIB"])
from fuzzywuzzy import fuzz  # noqa: E402
from rouge import Rouge  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LB16_DATA", str(Path.home() / "kvt_bench" / "data" / "longbench_full" / "data")))
TASKS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
         "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p")
NO_CHAT = ("trec", "triviaqa", "samsum", "lcc", "repobench-p")      # official pred.py: no chat template on these
FIRST_LINE = ("trec", "triviaqa", "samsum")                         # official eval.py scorer
_P = json.loads((HERE / "dataset2prompt.json").read_text(encoding="utf-8"))
_M = json.loads((HERE / "dataset2maxlen.json").read_text(encoding="utf-8"))
PROMPT = {t: _P[t] for t in TASKS}
MAXGEN = {t: int(_M[t]) for t in TASKS}
_CLASSES: dict[str, list[str]] = {}


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:                   # as in the official code (removes while iterating)
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:  # noqa: BLE001 - the official code returns 0 on any rouge failure
        return 0.0
    return scores["rouge-l"]["f"]


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(p), num_same / len(g)
    return (2 * precision * recall) / (precision + recall)


METRIC = {"narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score, "hotpotqa": qa_f1_score,
          "2wikimqa": qa_f1_score, "musique": qa_f1_score, "gov_report": rouge_score, "qmsum": rouge_score,
          "multi_news": rouge_score, "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
          "passage_retrieval_en": retrieval_score, "passage_count": count_score, "lcc": code_sim_score,
          "repobench-p": code_sim_score}


def all_classes(task: str) -> list[str]:
    """all_classes of a task (the same list on every row of the official data); [] when the task has none."""
    if task not in _CLASSES:
        with open(DATA / f"{task}.jsonl", encoding="utf-8") as f:
            _CLASSES[task] = json.loads(f.readline()).get("all_classes") or []
    return _CLASSES[task]


def score(task: str, pred: str, answers: list[str]) -> float:
    """official eval.py scorer for one prediction: a value in [0, 1]."""
    if task in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    cls = all_classes(task) if METRIC[task] is classification_score else None
    return max(float(METRIC[task](pred, a, all_classes=cls)) for a in answers)


def selftest() -> int:
    ok = True
    ok &= abs(qa_f1_score("The Eiffel Tower!", "eiffel tower") - 1.0) < 1e-9
    ok &= abs(qa_f1_score("Paris France", "Paris") - 2 / 3) < 1e-9
    ok &= retrieval_score("Paragraph 12", "Paragraph 12") == 1.0
    ok &= abs(retrieval_score("Paragraph 12 or 3", "Paragraph 12") - 0.5) < 1e-9
    ok &= count_score("there are 5 unique, maybe 6", "5") == 0.5
    ok &= classification_score("Type: Abbreviation", "Abbreviation", all_classes=["Abbreviation", "Entity"]) == 1.0
    ok &= classification_score("Entity or Abbreviation", "Entity", all_classes=["Abbreviation", "Entity"]) == 0.5
    ok &= code_sim_score("\n    return x + 1\n# c", "    return x + 1") == 1.0
    ok &= code_sim_score("# only a comment", "x") == 0.0
    ok &= abs(rouge_score("the cat sat on the mat", "the cat sat on the mat") - 1.0) < 1e-6
    ok &= rouge_score("", "reference") == 0.0
    ok &= len(TASKS) == 16 and all(t in PROMPT and t in MAXGEN and t in METRIC for t in TASKS)
    ok &= all("{context}" in PROMPT[t] for t in TASKS)
    print("SELFTEST", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else 2)
