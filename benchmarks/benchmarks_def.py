from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from jiwer import wer, cer

class Benchmarks:
    @staticmethod
    def evaluate_ner(true_entities, predicted_entities):
        """
        Evaluate Named Entity Recognition (NER) on extracted fields from audio transcription.
        Typical entities: location, date, time, plate, car model/brand, infraction type.

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
    def evaluate_stt(references, hypotheses):
        """
        Evaluate Speech-to-Text (STT) component by comparing reference and predicted transcriptions in Italian.
        Input: list of ground truth transcriptions and list of model-generated ones.

        args:
            references: list of ground truth transcriptions
            hypotheses: list of model-generated transcriptions

        returns:
            wer: Word Error Rate (WER) between references and hypotheses
            cer: Character Error Rate (CER) between references and hypotheses
        """
        assert len(references) == len(hypotheses), "STT inputs must have the same length"
        return {
            "wer": wer(references, hypotheses),
            "cer": cer(references, hypotheses),
        }

    @staticmethod
    def evaluate_cv(y_true, y_pred, average="macro", top_k_preds=None, y_true_top_k=None):
        """
        Evaluate computer vision classification for license plate, car brand, or model.
        Supports top-k evaluation for ambiguous or similar-looking car models.

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
