import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from benchmarks import Benchmarks

def test_ner_token_level():
    true = ["B-LOC", "O", "B-DATE", "I-DATE"]
    pred = ["B-LOC", "O", "B-DATE", "I-DATE"]
    metrics = Benchmarks.evaluate_ner(true, pred)
    assert metrics["f1"] == 1.0
    assert metrics["token_accuracy"] == 1.0

def test_ner_entity_level():
    true_seq = [["B-LOC", "O", "B-DATE", "I-DATE"]]
    pred_seq = [["B-LOC", "O", "B-DATE", "I-DATE"]]
    metrics = Benchmarks.evaluate_ner_entity_level(true_seq, pred_seq)
    assert metrics["f1"] == 1.0

def test_stt_metrics():
    references = ["il veicolo era parcheggiato in divieto di sosta"]
    hypotheses = ["il veicolo era parcheggiato a divieto di sosta"]
    metrics = Benchmarks.evaluate_stt(references, hypotheses)
    assert 0 <= metrics["wer"] <= 1
    assert 0 <= metrics["bleu_avg"] <= 1
    assert metrics["levenshtein_avg"] >= 0

def test_cv_metrics():
    y_true = ["fiat", "audi", "bmw"]
    y_pred = ["fiat", "bmw", "bmw"]
    top_k_preds = [["fiat", "opel"], ["bmw", "audi"], ["bmw", "mercedes"]]
    y_true_top_k = ["fiat", "audi", "bmw"]
    metrics = Benchmarks.evaluate_cv(y_true, y_pred, top_k_preds=top_k_preds, y_true_top_k=y_true_top_k)
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["top_k_accuracy"] == 1.0

def test_iou():
    pred_boxes = [[10, 20, 50, 80]]
    true_boxes = [[12, 22, 48, 78]]
    result = Benchmarks.evaluate_iou(pred_boxes, true_boxes)
    assert 0 <= result["iou_avg"] <= 1

def test_map():
    preds = [
        {
            "boxes": torch.tensor([[10, 20, 50, 80]], dtype=torch.float),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1])
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[12, 22, 48, 78]], dtype=torch.float),
            "labels": torch.tensor([1])
        }
    ]
    result = Benchmarks.evaluate_map(preds, targets)
    assert "map" in result
    assert result["map"] >= 0


#! To actually run the tests, you need to run the following command:
# pytest -v
# or 
# pytest test/test_benchmarks_pytest.py -v

# You can also run: python run_tests.py