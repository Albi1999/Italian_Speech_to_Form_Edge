from benchmarks import Benchmarks

def run_ner_evaluation():
    # Simulated NER output and reference
    true_entities = ["B-LOC", "O", "B-DATE", "I-DATE"]
    predicted_entities = ["B-LOC", "O", "B-DATE", "I-DATE"]

    metrics = Benchmarks.evaluate_ner(true_entities, predicted_entities)
    print("NER Metrics:", metrics)

def run_stt_evaluation():
    # Simulated STT output and reference
    references = ["il veicolo era parcheggiato in divieto di sosta"]
    hypotheses = ["il veicolo era parcheggiato a divieto di sosta"]

    metrics = Benchmarks.evaluate_stt(references, hypotheses)
    print("STT Metrics:", metrics)

def run_cv_evaluation():
    # Simulated CV output and reference
    y_true = ["fiat", "audi", "bmw"]
    y_pred = ["fiat", "bmw", "bmw"]
    top_k_preds = [["fiat", "opel"], ["bmw", "audi"], ["bmw", "mercedes"]]
    y_true_top_k = ["fiat", "audi", "bmw"]

    metrics = Benchmarks.evaluate_cv(y_true, y_pred, top_k_preds=top_k_preds, y_true_top_k=y_true_top_k)
    print("CV Metrics:", metrics)

if __name__ == "__main__":
    run_ner_evaluation()
    run_stt_evaluation()
    run_cv_evaluation()
