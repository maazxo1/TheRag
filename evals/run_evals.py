"""
Evaluation script for TheRaG pipeline.

Metrics:
  - Answer faithfulness : ROUGE-L between generated answer and ground truth
  - Confidence AUROC    : does high confidence correlate with correct answers?
  - Latency             : per-query end-to-end time
  - Retrieval recall    : does the correct context appear in retrieved chunks?

AUROC interpretation:
  - 0.50 = random (confidence is useless)
  - 0.70 = acceptable
  - 0.80+ = good — confidence reliably separates correct from incorrect answers

Usage:
    venv/Scripts/python evals/run_evals.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rouge_score import rouge_scorer as rs
from entrypoint.ingest import ingest_from_file
from entrypoint.query import run_pipeline
import config

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")
EVAL_SOURCE = os.path.join(os.path.dirname(__file__), "eval_source.md")

ROUGE_CORRECT_THRESHOLD = 0.25


def rouge_l(prediction: str, reference: str) -> float:
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """
    Compute AUROC from binary labels and continuous scores.
    Returns 0.5 if sklearn is unavailable or only one class present.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return float("nan")
        return roc_auc_score(labels, scores)
    except ImportError:
        # Manual trapezoidal AUROC
        paired = sorted(zip(scores, labels), reverse=True)
        tp = fp = 0
        prev_tp = prev_fp = 0
        auc = 0.0
        total_pos = sum(labels)
        total_neg = len(labels) - total_pos
        if total_pos == 0 or total_neg == 0:
            return float("nan")
        for _, label in paired:
            if label:
                tp += 1
            else:
                fp += 1
            tpr = tp / total_pos
            fpr = fp / total_neg
            prev_fpr = prev_fp / total_neg if total_neg else 0
            auc += (fpr - prev_fpr) * tpr
            prev_tp, prev_fp = tp, fp
        return auc


def retrieval_recall(stages: dict, ground_truth: str, top_k: int = 5) -> float:
    """Fraction of top_k hybrid-scored chunks that overlap with ground truth."""
    gt_words = set(ground_truth.lower().split())
    hybrid_scored = stages.get("hybrid_scored", [])
    hits = 0
    checked = 0
    for score, node in hybrid_scored[:top_k]:
        checked += 1
        chunk_words = set(node.get_content().lower().split())
        overlap = len(gt_words & chunk_words) / len(gt_words) if gt_words else 0
        if overlap >= 0.15:
            hits += 1
    return hits / checked if checked else 0.0


def run_evaluation():
    print("\n=== TheRaG Evaluation ===\n")

    with open(EVAL_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Ingesting eval source document: {os.path.basename(EVAL_SOURCE)}")
    index, bm25, child_nodes, parent_node_map = ingest_from_file(EVAL_SOURCE)
    print(f"Index ready: {len(child_nodes)} child chunks\n")

    results = []
    rouge_scores = []
    confidence_scores = []
    recall_scores = []
    latencies = []

    print(f"{'ID':<6} {'ROUGE-L':>8} {'Conf':>6} {'Recall':>7} {'Latency':>9}   Question")
    print("-" * 90)

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item["category"]

        result = run_pipeline(
            question=question,
            index=index,
            bm25=bm25,
            child_nodes=child_nodes,
            parent_node_map=parent_node_map,
            enable_multi_query=False,
            enable_hyde=False,
            enable_reranking=True,
        )

        answer = result["answer"]
        conf = result["confidence"]["composite"]
        latency = result["timings"]["total_ms"]
        rl = rouge_l(answer, ground_truth)
        recall = retrieval_recall(result["stages"], ground_truth)

        rouge_scores.append(rl)
        confidence_scores.append(conf)
        recall_scores.append(recall)
        latencies.append(latency)

        print(
            f"[{qid}] {rl:>7.3f} {conf:>6.3f} {recall:>7.2f} {latency:>8} ms"
            f"   {question[:55]}"
        )

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "rouge_l": round(rl, 3),
            "confidence": round(conf, 3),
            "retrieval_recall": round(recall, 3),
            "latency_ms": latency,
        })

    n = len(dataset)

    # AUROC: confidence predicting whether ROUGE-L >= threshold
    correct_labels = [1 if r >= ROUGE_CORRECT_THRESHOLD else 0 for r in rouge_scores]
    auroc = compute_auroc(correct_labels, confidence_scores)
    correct_count = sum(correct_labels)

    avg_rouge = sum(rouge_scores) / n
    avg_conf = sum(confidence_scores) / n
    avg_recall = sum(recall_scores) / n
    avg_latency = sum(latencies) / n

    print(f"\n{'='*60}")
    print(f"Questions evaluated   : {n}")
    print(f"Correct (ROUGE>={ROUGE_CORRECT_THRESHOLD})    : {correct_count}/{n}")
    print(f"Avg ROUGE-L           : {avg_rouge:.3f}  (faithfulness vs ground truth)")
    print(f"Avg retrieval recall  : {avg_recall:.3f}  (context overlap with GT)")
    print(f"Avg confidence        : {avg_conf:.3f}")
    if auroc != auroc:  # nan
        print(f"Confidence AUROC      : N/A (need both correct and incorrect answers)")
    else:
        auroc_label = "GOOD" if auroc >= 0.75 else ("ACCEPTABLE" if auroc >= 0.60 else "POOR")
        print(f"Confidence AUROC      : {auroc:.3f}  [{auroc_label}]")
    print(f"Avg latency           : {avg_latency:.0f} ms")
    print(f"{'='*60}\n")

    summary = {
        "total_questions": n,
        "correct_count": correct_count,
        "rouge_correct_threshold": ROUGE_CORRECT_THRESHOLD,
        "avg_rouge_l": round(avg_rouge, 3),
        "avg_retrieval_recall": round(avg_recall, 3),
        "avg_confidence": round(avg_conf, 3),
        "confidence_auroc": round(auroc, 3) if auroc == auroc else None,
        "avg_latency_ms": round(avg_latency),
    }

    output = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
