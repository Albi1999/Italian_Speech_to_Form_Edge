import os
import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Envirment variables
from dotenv import load_dotenv
load_dotenv()

# Noise and audio processing
from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine

# Azure
import azure.cognitiveservices.speech as speechsdk

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AzureSyntheticDatasetGenerator:
    def __init__(
        self,
        output_dir="data/synthetic_datasets/AzureTTS",
        num_samples=100,
        lang="it-IT", 
        voice_names=None,
        azure_key=None,
        azure_region=None,
        seed=42
    ):
        """
        Initializes the generator for Azure TTS.

        Args:
            output_dir (str): Directory to save generated audio and metadata.
            num_samples (int): Number of samples to generate.
            lang (str): Language code for synthesis (e.g., 'it-IT', 'en-US').
            voice_name (str): The specific Azure voice name to use
            azure_key (str): Azure Speech resource subscription key.
            azure_region (str): Azure Speech resource region.
            seed (int): Random seed for reproducibility.
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.lang = lang
        self.voice_name = voice_names
        self.seed = seed

        # Azure Credentials
        self.azure_key = azure_key
        self.azure_region = azure_region

        # Azure Speech Configuration
        try:
            self.speech_config = speechsdk.SpeechConfig(subscription=self.azure_key, region=self.azure_region)
            self.speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "azure_speech_sdk.log")
            logger.info(f"Azure Speech SDK configured for region: {self.azure_region}")
        except Exception as e:
            logger.error(f"Failed to initialize SpeechConfig: {e}")
            raise

        random.seed(seed)
        os.makedirs(self.output_dir, exist_ok=True)
        self.samples = []

        # Noise and Speaker Data
        self.noise_types = ["none", "white", "static", "hum", "street"]
        #self.speaker_types = ["original", "radio", "fast", "slow"]
        self.speaker_types = ["original"]
        self.street_noise_files = [
            os.path.join("data", "noise", "ambiance_traffic_people_on_the_street.wav"),
            os.path.join("data", "noise", "street-from-courtyard.mp3"),
            os.path.join("data", "noise", "street-raining.wav")
        ]

        # Sentence Generation Data
        self.cars = ["Fiat Panda", "Volkswagen Golf", "Renault Clio", "BMW Serie 1", "Opel Corsa", "Toyota Yaris", "Audi A3", "Mercedes Classe A",
                     "Ford Fiesta", "Peugeot 208", "Nissan Micra", "Kia Picanto", "Hyundai i20", "Seat Ibiza", "Skoda Fabia", "Dacia Sandero"]
        self.plates = ["AB123CD", "CD456EF", "GH789IJ", "KL321MN", "XY987ZT", "ZA456QP", "FG234LM", "TR098YU", "JK567OP", "UV123WX", "QR456ST", "EP789GH",
                       "CD123AB", "EF456GH", "IJ789KL", "MN321OP", "QR987ST", "UV654WX", "XY321ZA", "AB456CD", "EF789GH", "IJ123KL", "MN456OP",
                       "QR789ST", "UV321WX", "XY654ZA", "AB789CD", "EF123GH", "IJ456KL", "MN789OP", "QR321ST", "UV987WX", "XY123ZA"]
        self.agents = ["Mario Rossi", "Anna Bianchi", "Carlo Bruni", "Silvia Neri", "Luca Verdi", "Giulia Costa", "Marco Gentili", "Sara Bellini",
                       "Francesco Rizzo", "Elena Fontana", "Alessandro Moretti", "Chiara Romano", "Matteo Gallo"]
        self.streets = ["via del Corso, 12", "via Giuseppe Mazzini, 24", "via XX Settembre, 5",
                        "piazza Venezia, 1", "via Garibaldi, 11", "corso Buenos Aires, 98", "via Etnea, 45", "via Roma, 77",
                        "corso Vittorio Emanuele, 15", "via della Libertà, 3", "viale dei Giardini, 8", "via della Repubblica, 20"]
        self.violations = ["parcheggio in divieto di sosta", "parcheggio in doppia fila", "parcheggio su strisce pedonali",
                           "transito in zona accesso vietato", "parcheggio in zona rimozione", "parcheggio davanti a passo carrabile"]
        self.articles = ["158, 2", "7, 15", "157, 5", "6, 1a(1-12)", "9, 1", "14, 1", "185, 2", "3, 1", "4, 2", "5, 3", "6, 4", "7, 5", "8, 6"]
        self.vehicle_types = ["ciclomotore", "motoveicolo", "autovettura", "rimorchio", "macchina agricola", "macchina operatrice"]
        self.violation_types = ["civile", "penale", "stradale"]
        self.print_methods = ["bluetooth", "wifi"]
        self.print_languages = ["italiano"]
        self.print_options = ["stampa anche la comunicazione", "non stampare anche la comunicazione"]

    def generate_all(self):
        num_with_noise = 0
        num_clean = 0
        sentences = []
        generated_ids = set()

        # Ensure subdirectories exist
        os.makedirs(os.path.join(self.output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "sentences"), exist_ok=True)


        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            try:
                sample = self.generate_sample(i)
                if sample:
                    self.samples.append(sample)
                    generated_ids.add(sample["id"])
                    if sample["noise_type"] != "none":
                        num_with_noise += 1
                    else:
                        num_clean += 1
                    sentences.append(sample["text"])
                else:
                    logger.warning(f"Sample {i} generation failed and returned None.")
            except Exception as e:
                logger.error(f"Error generating sample {i}: {e}", exc_info=True)

        # Validation: Check if all samples were generated
        if len(generated_ids) != self.num_samples:
             logger.warning(f"Expected {self.num_samples} samples, but only {len(generated_ids)} were generated successfully.")
             missing_ids = set(range(self.num_samples)) - generated_ids
             logger.warning(f"Missing sample IDs: {sorted(list(missing_ids))}")

        self.save_metadata()
        self.save_sentences(sentences)

        logger.info("--- Generation Summary ---")
        logger.info(f"Total samples generated: {len(self.samples)}")
        logger.info(f"Samples with noise:    {num_with_noise}")
        logger.info(f"Clean samples:         {num_clean}")
        logger.info(f"Output directory:      {self.output_dir}")


    def generate_sample(self, idx):
        noise_file_used = None
        sentence, date_str, time_str = self.generate_sentence()
        speaker_type = random.choice(self.speaker_types)
        noise_type = random.choice(self.noise_types)
        selected_voice_name = random.choice(self.voice_names)
        logger.info(f"Sample {idx}: Using randomly selected voice '{selected_voice_name}'")

        audio_subdir = os.path.join(self.output_dir, "audio")
        base_path = os.path.join(audio_subdir, f"clean_{idx}.wav")
        final_path = os.path.join(audio_subdir, f"sample_{idx}.wav")

        # Azure Voice Synthesis
        try:
            #! SSML is needed to specify the exact voice #######################################################################################################
            ssml_string = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{self.lang}'>
                <voice name='{selected_voice_name}'> # Usa la voce selezionata
                    {sentence}
                </voice>
            </speak>
            """
            print(f"DEBUG: SSML String for sample {idx}:\n{ssml_string}")

            # Configure audio output to save to the base_path file
            audio_config = speechsdk.audio.AudioOutputConfig(filename=base_path)

            # Create the synthesizer
            # NOTE: SpeechSynthesizer should ideally be reused if possible for performance,
            # but creating it per sample is safer for handling potential state issues or errors.
            #! For large datasets, consider reusing it outside the loop.
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

            # Synthesize the text using SSML
            logger.info(f"Synthesizing sample {idx} to {base_path}...")
            result = synthesizer.speak_ssml_async(ssml_string).get()

            # Check the result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Successfully synthesized {base_path}")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(f"Speech synthesis canceled for sample {idx}: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"Cancellation Error Details: {cancellation_details.error_details}")
                # Delete the potentially empty/corrupt file
                if os.path.exists(base_path): os.remove(base_path)
                return None
            else:
                 logger.error(f"Unhandled synthesis result reason for sample {idx}: {result.reason}")
                 if os.path.exists(base_path): os.remove(base_path)
                 return None

            # Post-processing with Pydub (Noise/Effects)
            # Load the audio synthesized by Azure
            if not os.path.exists(base_path) or os.path.getsize(base_path) == 0:
                 logger.error(f"Azure synthesis completed but output file {base_path} is missing or empty for sample {idx}.")
                 return None

            clean_audio = AudioSegment.from_file(base_path)
            processed_audio = clean_audio # Start with the clean audio

            # Optional: Validate initial Azure audio (basic checks)
            # You might need different thresholds than gTTS
            # if len(processed_audio) < 1000 or processed_audio.dBFS < -50: # Example thresholds
            #    logger.warning(f"Azure audio {base_path} might be too short or silent.")
            #    # Decide if you want to raise an error or just log

            processed_audio = self.apply_voice_effect(processed_audio, speaker_type)

            # Apply noise
            if noise_type != "none":
                try:
                    noise = self.generate_noise(noise_type, len(processed_audio)) # Generate noise matching duration
                    if noise and len(noise) >= len(processed_audio):
                        if noise_type == "street" and hasattr(self, "last_noise_file"):
                             noise_file_used = os.path.basename(self.last_noise_file)
                        else:
                             noise_file_used = None

                        #! Overlay noise (adjust gain if needed, Azure voices might be louder/quieter)
                        #! You might want to normalize the clean audio *before* overlaying
                        # processed_audio = effects.normalize(processed_audio)
                        # noise = effects.normalize(noise).apply_gain(-15) # Example: make noise quieter
                        processed_audio = processed_audio.overlay(noise)
                        processed_audio = effects.normalize(processed_audio) # Normalize after overlay
                        logger.info(f"Applied noise '{noise_type}' to sample {idx}")

                    elif noise is None:
                         logger.warning(f"Noise generation failed for type '{noise_type}' on sample {idx}. Skipping noise.")
                         noise_type = "none"
                    else:
                         logger.warning(f"Generated noise duration mismatch for sample {idx} ({len(noise)}ms vs {len(processed_audio)}ms). Skipping noise.")
                         noise_type = "none"

                except Exception as e_noise:
                    logger.error(f"Error applying noise '{noise_type}' to sample {idx}: {e_noise}", exc_info=True)
                    noise_type = "none"


            # Export the final audio
            processed_audio.export(final_path, format="wav")
            logger.info(f"Exported final sample {idx} to {final_path}")

            # Final validation
            if len(processed_audio) < 1000:
                logger.warning(f"Final audio {final_path} is very short after processing.")

            return {
                "id": idx,
                "text": sentence,
                "audio_path": os.path.join("audio", os.path.basename(final_path)),
                "noise_type": noise_type,
                "has_noise": noise_type != "none",
                "noise_file": noise_file_used,
                "speaker_type": speaker_type,
                "azure_voice_used": selected_voice_name,
                "date": date_str,
                "time": time_str
            }

        except FileNotFoundError as e_fnf:
             logger.error(f"File not found during sample {idx} generation: {e_fnf}", exc_info=True)
             return None
        except Exception as e_main:
            logger.error(f"Unhandled error in generate_sample for idx {idx}: {e_main}", exc_info=True)
             # Clean up potentially created files if an error occurred mid-process
            if os.path.exists(base_path):
                try: os.remove(base_path)
                except OSError: pass
            if os.path.exists(final_path):
                try: os.remove(final_path)
                except OSError: pass
            return None # Indicate failure

    # Generate sentences for the dataset
    def generate_sentence(self):
        # Random selections for mandatory fields
        agent = random.choice(self.agents)
        vehicle_type = random.choice(self.vehicle_types)
        nationality = random.choice(['italia', 'spagna', 'svizzera', 'germania', 'austria', 'francia', 'slovenia', 'croazia', 'belgio', 'portogallo', 'grecia', 'olanda'])
        plate = random.choice(self.plates)
        kind_plate = random.choice(['ufficiale', 'speciale', 'prova', 'copertura'])
        street_info = random.choice(self.streets).rsplit(',', 1)
        street, civico = street_info[0].strip(), street_info[1].strip()
        violation = random.choice(self.violations)
        article_details = random.choice(self.articles).rsplit(', ', 1)
        article, comma = article_details[0], article_details[1] if len(article_details) > 1 else "(unknown)"
        violation_type = random.choice(self.violation_types)

        # Details specific to the vehicle
        cars = random.choice(self.cars)
        color = random.choice(['rosso', 'blu', 'verde', 'nero', 'grigio', 'giallo', 'bianco', 'viola', 'arancione'])
        chassis = f"{random.randint(10000000000000000, 99999999999999999)}"
        mass = f"{random.randint(1000, 4000)} kg"
        # Additional details
        print_method = random.choice(self.print_methods)
        print_language = random.choice(self.print_languages)
        print_option = random.choice(self.print_options)
        dt_violation = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        date_violation = dt_violation.strftime("%d/%m/%Y")
        time_violation = dt_violation.strftime("%H:%M")
        immediate_contestation = random.choice([True, False])

        templates = [
            f"Il giorno {date_violation}, alle ore {time_violation}, è stato redatto un verbale per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
            f"L'infrazione è avvenuta in {street}, al civico {civico}. È stata riscontrata una violazione del codice {violation_type}, articolo {article}, comma {comma} ovvero {violation}" +
            (f"Il verbale è stato eseguito sul posto, con contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Non è stato possibile contestare sul posto a causa dell'assenza del trasgressore. ") +
            f"Sono previsti {random.randint(1, 10)} punti da decurtare, e non è stata applicata alcuna sanzione accessoria. Stampa in {print_language} tramite {print_method}, {print_option}. ",

            f"In data {date_violation}, alle ore {time_violation}, è stato notificato un preavviso per un {vehicle_type} targato {plate} con targa {kind_plate}, immatricolato in {nationality}. " +
            f"L'infrazione è stata rilevata in {street}, al civico {civico}, con violazione del codice {violation_type}, articolo {article}, comma {comma}, in particolare si è riscontrato un {violation} " +
            "Non è stato possibile contestare per l'assenza del trasgressore, e per questo è indicato il motivo di mancata contestazione. " +
            f"Da decurtare ci sono {random.randint(1, 10)} punti. " +
            (f"Il veicolo è un {cars} di colore {color}, con numero di telaio {chassis} e massa {mass}. " if random.choice([True, False]) else "Ulteriori dettagli sul veicolo non sono necessari. ") +
            f"Necessito la stampa del preavviso tramite {print_method}, in {print_language}, {print_option}. ",

            f"Il {date_violation}, alle ore {time_violation}, sto emettendo una notifica di infrazione per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
            f"L'infrazione si è verificata presso {street}, al civico {civico}, violando il codice {violation_type}, in particolare l'articolo {article}, comma {comma} riguardante il {violation}. " +
            f"Devo decurtare {random.randint(1, 10)} punti. " +
            ("Applico una sanzione accessoria in base alle circostanze. " if random.choice([True, False]) else "Non è necessaria alcuna sanzione accessoria. ") +
            (f"Il {vehicle_type} è di colore {color}, con numero di telaio {chassis} e massa {mass}. " if random.choice([True, False]) else f"Non sono necessari ulteriori dettagli sul {vehicle_type}. ") +
            f"Stampamelo con il {print_method} in {print_language}, {print_option}. ",

            f"Alle {time_violation} del {date_violation}, sto redigendo un verbale per il veicolo {vehicle_type} targato {plate} con targa {kind_plate}, immatricolato in {nationality}" +
            f"L'infrazione è stata accertata in {street}, al civico {civico}, con violazione del codice {violation_type}, articolo {article},comma {comma} per {violation} " +
            f"Sono previsti {random.randint(1, 10)} punti da decurtare. " +
            (f"Contestazione immediata effettuata dall'agente {agent}, il verbale è stato redatto sul posto in {street}. " if immediate_contestation else "Non è stata possibile una contestazione immediata per l'assenza del trasgressore. ") +
            f"Non posso applicare alcuna sanzione accessoria. Stampa tramite {print_method}, in {print_language}, {print_option}. ",

            f"Verbale redatto in data {date_violation} alle ore {time_violation} per la violazione riscontrata al veicolo {vehicle_type}, marca {cars}, targato {plate} (tipo {kind_plate}) proveniente da {nationality}. " +
            f"In particolare, in {street}, civico {civico}, è stato accertato {violation}, in violazione dell'articolo {article}, comma {comma} del codice {violation_type}. " +
            (f"L'agente {agent} ha proceduto con contestazione immediata. " if immediate_contestation else "Data l'assenza del conducente, la contestazione non è stata immediata. ") +
            f"Si prevede la decurtazione di {random.randint(1, 10)} punti dalla patente. Richiedo la stampa in {print_language} via {print_method}, {print_option}. ",

            f"Si notifica che in data {date_violation}, alle ore {time_violation}, è stato emesso un preavviso di violazione per il veicolo {vehicle_type} di marca {cars}, con targa {plate} (tipo {kind_plate}), registrato in {nationality}. " +
            f"L'infrazione è avvenuta in {street}, al numero {civico}, e consiste in {violation}, come previsto dall'articolo {article}, comma {comma} del codice {violation_type}. " +
            "Non essendo stato possibile identificare il conducente sul posto, la notifica viene inviata al proprietario del veicolo. " +
            f"Sono previsti {random.randint(1, 10)} punti di decurtazione.  " +
            f"Si prega di stampare la presente notifica tramite {print_method}, in lingua {print_language}, {print_option}. ",

            f"Attenzione, sto generando un verbale per il veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per un'infrazione commessa il {date_violation} alle {time_violation}. " +
            f"L'infrazione è avvenuta in {street}, al numero {civico}, dove il veicolo era in {violation}.  L'articolo violato è il {article}, comma {comma} del codice {violation_type}. " +
            f"Verranno tolti {random.randint(1, 10)} punti dalla patente. " +
            f"Stampa questo verbale via {print_method} in {print_language}, {print_option}. ",

            f"Procedo con la stesura di un verbale per il veicolo {vehicle_type}, marca {cars}, targato {plate} (di tipo {kind_plate}), proveniente da {nationality}.  L'infrazione è stata rilevata il {date_violation} alle ore {time_violation}. " +
            f"Il veicolo si trovava in {street}, al civico {civico}, in {violation}, violando l'articolo {article}, comma {comma}, del codice {violation_type}. " +
            (f"L'agente {agent} ha provveduto alla contestazione immediata dell'infrazione. " if immediate_contestation else "Non è stato possibile effettuare la contestazione immediata a causa dell'assenza del trasgressore, quindi motivo mancata contestazione: assenza del trasgressore. ") +
            f"La violazione comporta la perdita di {random.randint(1, 10)} punti.  Richiedo la stampa del verbale in {print_language} tramite {print_method}, {print_option}. ",

            f"Emetto un avviso di violazione, quindi si tratta di un preavviso, per il veicolo {cars}, {vehicle_type}, targato {plate} con targa {kind_plate}, registrato in {nationality},  per un fatto accaduto il {date_violation} alle {time_violation}. " +
            f"Il veicolo si trovava in {street}, al civico {civico}, dove è stato riscontrato {violation}, in base all'articolo {article}, comma {comma}, del codice {violation_type}. " +
            f"Questa infrazione comporta una decurtazione di {random.randint(1, 10)} punti.  " +
            f"Genera la stampa dell'avviso via {print_method} in {print_language}, {print_option}. ",

            f"Redazione verbale in corso per il veicolo {vehicle_type} di marca {cars}, con targa {plate} (tipo {kind_plate}), proveniente da {nationality}.  Fatto avvenuto il {date_violation} alle ore {time_violation}. " +
            f"Posizione: {street}, numero civico {civico}. Infrazione: {violation}, in violazione dell'articolo {article}, comma {comma}, del codice {violation_type}. " +
            f"Decurtazione prevista: {random.randint(1, 10)} punti. " +
            (f"Si applica contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Impossibile contestare immediatamente per assenza del trasgressore, motivo: assenza del trasgressore. ") +
            f"Stampa necessaria tramite {print_method} in {print_language}, {print_option}. ",

            f"Sto emettendo un verbale per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
            f"L'infrazione è avvenuta in {street}, civico {civico}: {violation}. Si applica l'articolo {article}, comma {comma} del codice {violation_type}. " +
            f"A seguito della violazione verranno decurtati {random.randint(1, 10)} punti. " +
            (f"Il verbale è stato contestato immediatamente dall'agente {agent}. " if immediate_contestation else "A causa dell'assenza del trasgressore non è stato possibile effettuare la contestazione immediata. ") +
            f"Si prega di procedere con la stampa in {print_language} tramite {print_method}, {print_option}. ",

            f"Si notifica un preavviso relativo al veicolo {vehicle_type} targato {plate} di tipo {kind_plate} proveniente da {nationality}, marca {cars}. " +
            f"La violazione, ovvero {violation}, si è verificata in data {date_violation} alle ore {time_violation} in {street}, al numero {civico}. " +
            f"L'infrazione rientra nell'articolo {article}, comma {comma} del codice {violation_type}. " +
            (f"Contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Non è stato possibile contestare immediatamente a causa dell'assenza del trasgressore, motivo mancata contestazione: assenza del trasgressore. ") +
            f"La sanzione prevede la perdita di {random.randint(1, 10)} punti. Si richiede la stampa in lingua {print_language}, tramite {print_method}, {print_option}. ",

            f"Sto procedendo con la redazione di un verbale per {vehicle_type}, marca {cars}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. " +
            f"In data {date_violation} alle ore {time_violation}, in {street}, al civico {civico}, è stata commessa la seguente violazione: {violation} (articolo {article}, comma {comma} del codice {violation_type}). " +
            f"Dalla violazione consegue la perdita di {random.randint(1, 10)} punti. " +
            f"Stampa in modalità {print_method} e in lingua {print_language}, {print_option}. ",

            f"Si avvisa che è in corso la stesura di un preavviso di violazione per il veicolo {vehicle_type} con targa {plate} di tipo {kind_plate}, marca {cars}, proveniente da {nationality}. " +
            f"L'infrazione, nello specifico {violation}, si è verificata in data {date_violation} alle ore {time_violation} in {street}, al numero {civico} (articolo {article}, comma {comma} del codice {violation_type}). " +
            f"Si comunica che dalla violazione conseguirà una decurtazione di {random.randint(1, 10)} punti. " +
            f"Si richiede la stampa in lingua {print_language}, modalità {print_method}, {print_option}. ",

            f"In data {date_violation}, alle ore {time_violation}, si eleva verbale al veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per violazione dell'articolo {article}, comma {comma} del codice {violation_type}, ovvero {violation}, rilevata in {street}, al civico {civico}. " +
            (f"La contestazione è stata immediata, effettuata dall'agente {agent}. " if immediate_contestation else f"Non è stato possibile effettuare la contestazione immediata, motivo: assenza del trasgressore. ") +
            f"Dalla violazione derivano {random.randint(1, 10)} punti di decurtazione. Stampa necessaria in lingua {print_language} via {print_method}, {print_option}. ",

            f"Emissione di preavviso per il veicolo {vehicle_type}, marca {cars}, targato {plate} di tipo {kind_plate}, proveniente da {nationality}. In data {date_violation} alle ore {time_violation}, è stata rilevata la violazione di {violation} in {street}, al numero {civico}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
            f"Per tale infrazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa del preavviso tramite {print_method} in lingua {print_language}, {print_option}. ",

            f"Si sta redigendo verbale per violazione accertata il {date_violation} alle ore {time_violation} in {street}, al civico {civico}, a carico del veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione contestata è {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
            (f"La contestazione è avvenuta immediatamente da parte dell'agente {agent}. " if immediate_contestation else f"Non è stato possibile contestare immediatamente, motivo: assenza del trasgressore. ") +
            f"Sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa in lingua {print_language} via {print_method}, {print_option}. ",

            f"Si notifica la presenza di un preavviso di violazione per il veicolo {vehicle_type}, marca {cars}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione è avvenuta il {date_violation} alle ore {time_violation} in {street}, al numero civico {civico}, ed è relativa a {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
            f"Per tale violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa di tale preavviso in lingua {print_language} tramite {print_method}, {print_option}. ",

            f"Sto emettendo un verbale per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per infrazione commessa in data {date_violation}, ore {time_violation}. In {street}, civico {civico}, è stato rilevato {violation}, ai sensi dell'articolo {article}, comma {comma}, del codice {violation_type}. " +
            (f"Contestazione immediata eseguita dall'agente {agent}. " if immediate_contestation else f"Impossibile effettuare la contestazione immediata, motivo: assenza del trasgressore. ") +
            f"La violazione comporta la decurtazione di {random.randint(1, 10)} punti. Richiesta stampa in {print_language} via {print_method}, {print_option}. ",

            f"Si notifica un preavviso per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, proveniente da {nationality}. Il giorno {date_violation}, alle ore {time_violation}, in {street}, al numero civico {civico}, è stata commessa l'infrazione di {violation}, come previsto dall'articolo {article}, comma {comma}, del codice {violation_type}. " +
            f"Per questa violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Chiedo la stampa del preavviso in {print_language} via {print_method}, {print_option}. ",
        ]

        selected_sentence = random.choice(templates)
        return selected_sentence, date_violation, time_violation


    # Generate Noise
    def generate_noise(self, noise_type, duration_ms, volume_db=-10.0):
        """Generates a noise segment of a specific type and duration."""
        self.last_noise_file = None # Reset last noise file
        try:
            if noise_type == "white":
                return WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(volume_db)

            elif noise_type == "static":
                freqs = [1000, 1200, 1500]
                waves = [Sine(f).to_audio_segment(duration=duration_ms).apply_gain(volume_db - 5) for f in freqs]
                combined_noise = waves[0]
                for wave in waves[1:]:
                   combined_noise = combined_noise.overlay(wave)
                return combined_noise

            elif noise_type == "hum":
                return Sine(60).to_audio_segment(duration=duration_ms).apply_gain(volume_db)

            elif noise_type == "street":
                if not self.street_noise_files:
                     logger.warning("Street noise requested, but no street noise files are configured.")
                     return None
                noise_path = random.choice(self.street_noise_files)
                self.last_noise_file = noise_path

                if not os.path.exists(noise_path):
                    logger.error(f"Street noise file missing: {noise_path}")
                    return None

                noise = AudioSegment.from_file(noise_path)

                if len(noise) == 0:
                    logger.warning(f"Street noise file {noise_path} loaded as empty segment.")
                    return None
                if len(noise) < duration_ms:
                    repeat_count = (duration_ms // len(noise)) + 1
                    noise *= repeat_count
                    logger.debug(f"Repeating street noise {repeat_count} times.")


                # Trim to exact duration and apply gain
                return noise[:duration_ms].apply_gain(volume_db)

            elif noise_type == "none":
                 return None
            else:
                logger.warning(f"Unknown noise type requested: {noise_type}")
                return None

        except Exception as e:
            logger.error(f"Error generating noise type '{noise_type}': {e}", exc_info=True)
            return None


    # Apply voice effect
    def apply_voice_effect(self, audio, speaker_type):
        """Applies pydub-based effects to simulate speaker variations."""
        logger.debug(f"Applying effect '{speaker_type}'")
        try:
            if speaker_type == "fast":
                # Speed up, which also increases pitch
                return audio.speedup(playback_speed=1.2)
            elif speaker_type == "slow":
                 # Slow down, which also decreases pitch
                return audio.speedup(playback_speed=0.85)
            elif speaker_type == "radio":
                # Apply band-pass filter (approximate) and slight gain reduction
                filtered_audio = effects.high_pass_filter(audio, cutoff=300)
                filtered_audio = effects.low_pass_filter(filtered_audio, cutoff=3000)
                return filtered_audio.apply_gain(-3)
            elif speaker_type == "original":
                 return audio # No change
            else:
                 logger.warning(f"Unknown speaker type effect '{speaker_type}'. Returning original audio.")
                 return audio # Return original if type is unknown
        except Exception as e:
             logger.error(f"Error applying voice effect '{speaker_type}': {e}", exc_info=True)
             return audio # Return original audio on error


    # Save Metadata
    def save_metadata(self):
        metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "samples.json")

        try:
             with open(metadata_path, "w", encoding="utf-8") as f:
                 json.dump(self.samples, f, indent=2, ensure_ascii=False)
             logger.info(f"Metadata saved to: {metadata_path}")
        except IOError as e:
             logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        except TypeError as e:
             logger.error(f"Failed to serialize metadata to JSON: {e}")


    # Save Sentences
    def save_sentences(self, sentences):
        sentence_dir = os.path.join(self.output_dir, "sentences")
        os.makedirs(sentence_dir, exist_ok=True)
        sentences_path = os.path.join(sentence_dir, "sentences.json")

        try:
            with open(sentences_path, "w", encoding="utf-8") as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
            logger.info(f"Sentences saved to: {sentences_path}")
        except IOError as e:
            logger.error(f"Failed to save sentences to {sentences_path}: {e}")



if __name__ == "__main__":
    NUM_SAMPLES = 10
    OUTPUT_BASE = "data/synthetic_datasets/Azure"
    LANG = "it-IT"

    cognitive_svc_key = os.getenv("COGNITIVE_SERVICE_KEY")
    cognitive_svc_location = os.getenv("COGNITIVE_SERVICE_LOCATION")

    azure_us_key = os.getenv("AZURE_US_KEY")
    azure_us_location = os.getenv("AZURE_US_LOCATION")

    azure_eu_key = os.getenv("AZURE_EU_KEY")
    azure_eu_location = os.getenv("AZURE_EU_LOCATION")

    # Validate Credentials are Loaded
    credentials_loaded = True
    if not cognitive_svc_key or not cognitive_svc_location:
        logger.error("COGNITIVE_SERVICE_KEY or COGNITIVE_SERVICE_LOCATION not found in environment (check .env file).")
        credentials_loaded = False
    if not azure_us_key or not azure_us_location:
        logger.error("AZURE_US_KEY or AZURE_US_LOCATION not found in environment (check .env file).")
        credentials_loaded = False
    if not azure_eu_key or not azure_eu_location:
        logger.error("AZURE_EU_KEY or AZURE_EU_LOCATION not found in environment (check .env file).")
        credentials_loaded = False

    # Store credentials in a dictionary
    credentials = {
        "standard": {"key": cognitive_svc_key, "region": cognitive_svc_location},
        "us_openai": {"key": azure_us_key, "region": azure_us_location},
        "eu_openai": {"key": azure_eu_key, "region": azure_eu_location},
    }

    # Define Voice Configurations
    #! REMEMBER: Replace placeholder OpenAI voice names with ACTUAL Italian voice names
    #TODO: Find for voices names
    voice_configs = {
         # Standard Cognitive Services Voice
        "standard_voices": {
            "voice_names": ["it-IT-DiegoNeural", "it-IT-ElsaNeural"], # Lista di voci
            "credential_set": "standard",
            "output_subdir": "Azure_Standard_Italian"
        },
        # Azure OpenAI Voices
        "openai_gpt4o_mini": {
            #! VERIFY this name
            "voice_names": ["azure://eastus.azure.openai.com/openai/voice/preview-gpt-4o-mini-it-IT-Studio-C"],
            "credential_set": "us_openai",
            "output_subdir": "OpenAI_GPT4oMini"
        },
        "openai_tts": {
            #! VERIFY this name
            "voice_names": ["en-US-NovaMultilingualNeural", "en-US-OnyxMultilingualNeural"],
            "credential_set": "eu_openai",
            "output_subdir": "OpenAI_TTS"
        },
        "openai_tts_hd": {
             #! VERIFY this name
            "voice_names": ["azure://swedencentral.azure.openai.com/openai/voice/tts-1-hd-it-IT-Studio-E"],
            "credential_set": "eu_openai",
            "output_subdir": "OpenAI_TTS_HD"
        },
    }

    # Generate Datasets for Each Voice
    if not credentials_loaded:
        logger.error("One or more Azure credential sets are missing. Cannot proceed.")
    else:
        logger.info("\nStarting Dataset Generation...")
        for config_name, params in voice_configs.items():
            logger.info(f"\nGenerating dataset for: {config_name}")
            output_directory = os.path.join(OUTPUT_BASE, params["output_subdir"])

            cred_set_name = params.get("credential_set")
            if not cred_set_name or cred_set_name not in credentials:
                logger.error(f"Invalid or missing 'credential_set' ('{cred_set_name}') for voice config: {config_name}. Skipping.")
                continue

            current_credentials = credentials[cred_set_name]
            current_key = current_credentials.get("key")
            current_region = current_credentials.get("region")

            if not current_key or not current_region:
                 logger.error(f"Credentials key or region missing for set '{cred_set_name}' used by {config_name}. Skipping.")
                 continue

            logger.info(f"Using credentials from set: '{cred_set_name}' (Region: {current_region})")

            try:
                generator = AzureSyntheticDatasetGenerator(
                    output_dir=output_directory,
                    num_samples=NUM_SAMPLES,
                    lang=LANG,
                    voice_name=params["voice_names"],
                    azure_key=current_key,
                    azure_region=current_region,
                    seed=42
                )
                generator.generate_all()
                logger.info(f"Finished dataset for: {config_name}")

            except ValueError as ve:
                 logger.error(f"Configuration error for {config_name}: {ve}")
            except speechsdk.exceptions.ConnectionFailureException as cfe:
                 logger.error(f"Connection Failure for {config_name} (Region: {current_region}): {cfe}. Check region, key, and network for credential set '{cred_set_name}'.")
            except Exception as ex:
                logger.error(f"Failed to generate dataset for {config_name}: {ex}", exc_info=True)

        logger.info("\nAll Dataset Generation Tasks Complete")