from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from jiwer import wer, cer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from seqeval.metrics import classification_report, precision_score as seq_precision, recall_score as seq_recall, f1_score as seq_f1
import Levenshtein

class Benchmarks:
    @staticmethod
    def evaluate_ner(true_entities, predicted_entities):
        """
        Token-level evaluation for NER.

        args:
            true_entities: list of true entities
            predicted_entities: list of predicted entities

        returns:
            precision: fraction of correct predictions among all predicted entities
            recall: fraction of correct predictions among all true entities
            f1: harmonic mean of precision and recall
            token_accuracy: fraction of correct predictions among all tokens
        """
        return {
            "precision": precision_score(true_entities, predicted_entities, average='micro', zero_division=0),
            "recall": recall_score(true_entities, predicted_entities, average='micro', zero_division=0),
            "f1": f1_score(true_entities, predicted_entities, average='micro', zero_division=0),
            "token_accuracy": accuracy_score(true_entities, predicted_entities)
        }
    
    @staticmethod
    def evaluate_ner_entity_level(true_entities_seq, predicted_entities_seq):
        """
        Entity-level evaluation using seqeval.

        args:
            true_entities_seq: list of true entity sequences
            predicted_entities_seq: list of predicted entity sequences

        returns:
            precision: fraction of correct predictions among all predicted entities
            recall: fraction of correct predictions among all true entities
            f1: harmonic mean of precision and recall
            report: classification report with precision, recall, and f1 scores
        """
        return {
            "precision": seq_precision(true_entities_seq, predicted_entities_seq),
            "recall": seq_recall(true_entities_seq, predicted_entities_seq),
            "f1": seq_f1(true_entities_seq, predicted_entities_seq),
            "report": classification_report(true_entities_seq, predicted_entities_seq, digits=3)
        }

    @staticmethod
    def evaluate_stt(references, hypotheses):
        """
        Evaluate STT using WER, CER, BLEU, ROUGE, and Levenshtein.

        args:
            references: list of ground truth transcriptions
            hypotheses: list of model-generated transcriptions

        returns:
            wer: word error rate
            cer: character error rate
            bleu_avg: average BLEU score
            rouge1_avg_f1: average ROUGE-1 F1 score
            rougeL_avg_f1: average ROUGE-L F1 score
            levenshtein_avg: average Levenshtein distance
        """
        assert len(references) == len(hypotheses), "STT inputs must have the same length"

        smooth = SmoothingFunction().method1
        bleu_scores = [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth) for ref, hyp in zip(references, hypotheses)]

        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]

        levenshtein_scores = [Levenshtein.distance(ref, hyp) for ref, hyp in zip(references, hypotheses)]

        return {
            "wer": sum([wer(r, h) for r, h in zip(references, hypotheses)]) / len(references),
            "cer": sum([cer(r, h) for r, h in zip(references, hypotheses)]) / len(references),
            "bleu_avg": sum(bleu_scores) / len(bleu_scores),
            "rouge1_avg_f1": sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
            "rougeL_avg_f1": sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores),
            "levenshtein_avg": sum(levenshtein_scores) / len(levenshtein_scores),
        }

    @staticmethod
    def evaluate_cv(y_true, y_pred, average="macro", top_k_preds=None, y_true_top_k=None):
        """
        Classification evaluation (top-k optional).

        args:
            y_true: list of true labels
            y_pred: list of predicted labels
            average: averaging strategy for precision, recall, and f1 scores
            top_k_preds: list of top-k predicted labels
            y_true_top_k: list of true labels for top-k evaluation

        returns:
            accuracy: fraction of correct predictions among all samples
            precision: fraction of correct predictions among all predicted labels
            recall: fraction of correct predictions among all true labels
            f1: harmonic mean of precision and recall
            top_k_accuracy: fraction of correct predictions among all samples for top-k evaluation
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        if top_k_preds and y_true_top_k:
            top_k_accuracy = sum([
                1 if true in top_k else 0 
                for true, top_k in zip(y_true_top_k, top_k_preds)
            ]) / len(y_true_top_k)
            metrics["top_k_accuracy"] = top_k_accuracy

        return metrics

    @staticmethod
    def evaluate_iou(pred_boxes, true_boxes):
        """
        Compute intersection over union (IoU) for object detection.

        args:
            pred_boxes: list of predicted bounding boxes
            true_boxes: list of true bounding boxes

        returns:
            iou_avg: average IoU score
        """
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea)

        iou_scores = [iou(p, t) for p, t in zip(pred_boxes, true_boxes)]
        return {
            "iou_avg": sum(iou_scores) / len(iou_scores)
        }

    @staticmethod
    def evaluate_map(preds, targets):
        """
        Compute mean average precision (mAP) for object detection.

        args:
            preds: list of predicted bounding boxes, scores, and labels
            targets: list of true bounding boxes and labels

        returns:
            mAP: mean average precision
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision #! Moved here to avoid problems with onnxruntime and torchmetrics
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        return metric.compute()