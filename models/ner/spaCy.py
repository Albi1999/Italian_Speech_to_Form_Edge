import spacy
import time
import os

class SpacyNERExtractor:
    """
    Class to extract Named Entities using spaCy Italian models.
    """
    def __init__(self, model_name="it_core_news_sm"):
        """
        Initializes the NER extractor with a specific spaCy model.

        Args:
            model_name (str): The name of the spaCy Italian model to load.
                                Should be one of:
                                        "it_core_news_sm",
                                        "it_core_news_md",
                                        "it_core_news_lg".
        """
        self.model_name = model_name
        self.name = f"spaCy NER ({model_name})"

        # Try to load the model
        try:
            # Disable unnecessary components for faster NER inference since only NER is needed
            self.nlp = spacy.load(self.model_name, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            print(f"Successfully loaded spaCy model: {self.model_name}")
        except OSError:
            print(f"Error: spaCy model '{self.model_name}' not found.")
            print(f"Please download it by running: python -m spacy download {self.model_name}")
            self.nlp = None # Set nlp to None if loading failed

    def _format_result(self, text_input, entities, duration):
        """
        Formats the NER results.
        """
        return {
            "model": self.name,
            "input_text": text_input,
            "entities": entities, # List of dictionaries for each entity
            "time": duration,
        }

    def extract_entities(self, text):
        """
        Extracts named entities from the input text.

        Args:
            text (str): The text to process for NER.

        Returns:
            dict: A dictionary containing the model name, input text,
                  a list of extracted entities, and the processing time.
                  Returns None if the model failed to load.
        """
        if not self.nlp:
            print(f"Model '{self.model_name}' was not loaded successfully. Cannot extract entities.")
            # Return a result indicating failure
            return self._format_result(text, [], 0.0)

        start_time = time.time()

        # Process the text with spaCy
        doc = self.nlp(text)

        # Extract entities
        extracted_entities = []
        for ent in doc.ents:
            extracted_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

        duration = time.time() - start_time

        return self._format_result(text, extracted_entities, duration)

# --- Example Usage ---
if __name__ == '__main__':
    # Copy and run the following commands in your terminal to download the models:
    # python -m spacy download it_core_news_sm
    # python -m spacy download it_core_news_md
    # python -m spacy download it_core_news_lg

    # Example text
    sample_text_1 = "il giorno due aprile duemilaventicinque alle otto meno dieci è stato redatto un verbale per il veicolo audi a tre molto veicolo immatricolati in olanda con targa kazaka quattrocentocinquanta sei cupi di tipo speciale l'infrazione e avvenuta in viale dei giardini al civico e stata riscontrata una violazione del codice penale articolo centottantacinque comma due ovvero parcheggio davanti a passo carabine non è stato possibile contestare sul posto a causa dell'assenza del trasgressori sono previsti tre punti da decurtare e non è stata applicata alcuna sanzione accessoria stampa in italiano tramite vuoi fai non stampare anche la comunicazione"
    sample_text_2 = "si avvisa che è in corso la stesura di un preavviso di violazione per il veicolo rimorchio con targa i centoventitre chi lo litri di tipo ufficiale marca fox vaga in golf proveniente da grecia l'infrazione nello specifico parcheggia in doppia fila sia e verificata in data diciotto marzo duemilaventicinque alle ore ventuno e ventitré in via della repubblica al numero venti articolo quattordici con ma uno del codice civile si comunica che dalla violazione conseguirà una decurtazione di otto punti si richiede la stampa in lingua italiano modalità hawaii fai non stampare anche la comunicazione"


    models_to_test = ["it_core_news_sm", "it_core_news_md", "it_core_news_lg"]
    texts_to_test = [sample_text_1, sample_text_2]

    for model_name in models_to_test:
        print(f"\n--- Testing Model: {model_name} ---")
        ner_extractor = SpacyNERExtractor(model_name=model_name)

        if ner_extractor.nlp: # Check if model loaded successfully
            for i, text in enumerate(texts_to_test):
                print(f"\n--- Processing Text {i+1} ---")
                result = ner_extractor.extract_entities(text)
                print(f"Model: {result['model']}")
                print(f"Time taken: {result['time']:.4f} seconds")
                print("Extracted Entities:")
                if result['entities']:
                    for entity in result['entities']:
                        print(f"  - Text: '{entity['text']}', Label: {entity['label']}, Start: {entity['start_char']}, End: {entity['end_char']}")
                else:
                    print("  No entities found.")
        print("-" * (len(model_name) + 18))