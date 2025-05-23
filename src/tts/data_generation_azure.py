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
        seed=456734
    ):
        """
        Initializes the generator for Azure Standard TTS voices.

        Args:
            output_dir (str): Directory to save generated audio and metadata.
            num_samples (int): Number of samples to generate.
            lang (str): Language code for synthesis (e.g., 'it-IT').
            voice_names (list): List of standard Azure voice names to randomly choose from.
            azure_key (str): Azure Cognitive Services subscription key.
            azure_region (str): Azure Cognitive Services resource region.
            seed (int): Random seed for reproducibility.
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.lang = lang
        self.voice_names = voice_names
        if not self.voice_names:
             raise ValueError("voice_names list cannot be empty")
        self.seed = seed

        # Azure Credentials (Cognitive Services)
        self.azure_key = azure_key
        self.azure_region = azure_region

        # Azure Speech Configuration
        try:
            self.speech_config = speechsdk.SpeechConfig(subscription=self.azure_key, region=self.azure_region)
            self.speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "azure_speech_sdk.log")
            logger.info(f"Azure Speech SDK configured for region: {self.azure_region}")
            logger.info(f"Available standard voices for random selection: {len(self.voice_names)}")
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
                     "Ford Fiesta", "Peugeot 208", "Nissan Micra", "Kia Picanto", "Hyundai i20", "Seat Ibiza", "Skoda Fabia", "Dacia Sandero",
                     "Citroen C3", "Mazda 2", "Honda Jazz", "Subaru Impreza", "Mitsubishi Colt", "Suzuki Swift", "Chevrolet Spark",
                     "Lancia Ypsilon", "Alfa Romeo Giulietta", "Fiat 500", "Volkswagen Polo", "Renault Captur", "Peugeot 2008", "Nissan Juke",
                     "Kia Stonic", "Hyundai Kona", "Seat Arona", "Skoda Kamiq", "Dacia Duster", "Citroen C4 Cactus", "Mazda CX-3", "Honda HR-V",
                     "Subaru XV", "Mitsubishi ASX", "Suzuki Vitara", "Chevrolet Trax", "Lancia Delta", "Alfa Romeo Stelvio", "Fiat Tipo",
                     "Ducati Monster", "Yamaha MT-07", "Kawasaki Z650", "Honda CB650R", "BMW F 900 R", "Suzuki SV650", "Triumph Street Triple",
                     "KTM 790 Duke", "Harley-Davidson Street 750", "Moto Guzzi V7", "Royal Enfield Interceptor 650", "Benelli Leoncino 500",
                     "Kawasaki Ninja 650", "Yamaha YZF-R7", "Honda CBR650R", "BMW F 900 XR", "Ducati Multistrada V2", "Triumph Tiger 850 Sport"]
        self.plates = ["AB123CD", "CD456EF", "GH789IJ", "KL321MN", "XY987ZT", "ZA456QP", "FG234LM", "TR098YU", "JK567OP", "UV123WX", "QR456ST", "EP789GH",
                       "CD123AB", "EF456GH", "IJ789KL", "MN321OP", "QR987ST", "UV654WX", "XY321ZA", "AB456CD", "EF789GH", "IJ123KL", "MN456OP",
                       "QR789ST", "UV321WX", "XY654ZA", "AB789CD", "EF123GH", "IJ456KL", "MN789OP", "QR321ST", "UV987WX", "XY123ZA", "AB654CD",
                       "EF321GH", "IJ654KL", "MN123OP", "QR456ST", "UV789WX", "XY987ZA", "AB321CD", "EF654GH", "IJ789KL", "MN456OP", "QR123ST",
                       "UV456WX", "XY321ZA", "AB987CD", "EF123GH", "IJ456KL", "MN789OP", "QR654ST", "UV321WX", "XY987ZA", "AB123CD", "EF456GH",
                       "IJ789KL", "MN321OP", "QR987ST", "UV654WX", "XY321ZA", "AB456CD", "EF789GH", "IJ123KL", "MN456OP", "QR789ST", "UV321WX"]
        self.agents = ["Mario Rossi", "Anna Bianchi", "Carlo Bruni", "Silvia Neri", "Luca Verdi", "Giulia Costa", "Marco Gentili", "Sara Bellini", 
                       "Francesco Rizzo", "Elena Fontana", "Alessandro Moretti", "Chiara Romano", "Matteo Gallo"]
        self.streets = ["via del Corso, 12", "via Giuseppe Mazzini, 24", "via XX Settembre, 5",
                        "piazza Venezia, 1", "via Garibaldi, 11", "corso Buenos Aires, 98", "via Etnea, 45", "via Roma, 77",
                        "corso Vittorio Emanuele, 15", "via della Libertà, 3", "viale dei Giardini, 8", "via della Repubblica, 20"
                        "via dei Fori Imperiali, 10", "viale della Stazione, 4", "via della Storia, 6", "corso Italia, 30",
                        "via della Cultura, 9", "viale della Scienza, 14", "via della Musica, 18", "corso della Storia, 22",
                        "via della Pace, 2", "viale della Vittoria, 9", "via della Concordia, 14", "corso della Libertà, 7",
                        "via della Speranza, 18", "viale della Libertà, 22", "via della Giustizia, 16", "corso della Repubblica, 19",
                        "via della Libertà, 21", "viale della Concordia, 13", "via della Speranza, 17", "corso della Giustizia, 23"]
        self.violations = ["parcheggio in divieto di sosta", "parcheggio in doppia fila", "parcheggio su strisce pedonali",
                           "transito in zona accesso vietato", "parcheggio in zona rimozione", "parcheggio davanti a passo carrabile"]
        self.articles = ["158, 2", "7, 15", "157, 5", "6, 1a(1-12)", "9, 1", "14, 1", "185, 2", "3, 1", "4, 2", "5, 3", "6, 4", "7, 5", "8, 6",
                         "9, 7", "10, 8", "121, 9", "12, 10", "13, 11", "14, 12", "3415, 13", "16, 14", "17, 15", "1228, 16", "19, 17", "20, 18",
                         "21, 19", "22, 20", "23, 21", "24, 22", "2235, 23", "26, 24", "27, 25", "28, 26", "29, 27", "30, 28", "31, 29"
                         "32, 30", "33, 31", "34, 32", "35, 33", "36, 34", "37, 35", "38, 36", "39, 37", "40, 38", "41, 39", "42, 40",
                         "43, 41", "44, 42", "45, 43", "46, 44", "47, 45", "48, 46", "49, 47", "50, 48", "51, 49", "52, 50", "53, 51"]
        self.vehicle_types = ["ciclomotore", "motoveicolo", "autovettura", "rimorchio", "macchina agricola", "macchina operatrice"]
        self.violation_types = ["civile", "penale", "stradale"]
        self.print_methods = ["bluetooth", "wifi"]
        self.print_languages = ["italiano", "inglese", "francese", "spagnolo", "tedesco", "portoghese", "olandese"]
        self.print_options = ["stampa anche la comunicazione", "non stampare anche la comunicazione"]

    def generate_all(self):
        num_with_noise = 0
        num_clean = 0
        sentences = []
        generated_ids = set()

        # Ensure subdirectories exist
        audio_dir = os.path.join(self.output_dir, "audio")
        metadata_dir = os.path.join(self.output_dir, "meta")
        sentences_dir = os.path.join(self.output_dir, "tx")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(sentences_dir, exist_ok=True)

        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            try:
                sample = self.generate_sample(i, audio_dir)
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

        # Validation
        if len(generated_ids) != self.num_samples:
             logger.warning(f"Expected {self.num_samples} samples, but only {len(generated_ids)} were generated successfully.")
             missing_ids = set(range(self.num_samples)) - generated_ids
             logger.warning(f"Missing sample IDs: {sorted(list(missing_ids))}")

        self.save_metadata(metadata_dir)
        self.save_sentences(sentences, sentences_dir)

        logger.info("--- Generation Summary ---")
        logger.info(f"Total samples generated: {len(self.samples)}")
        logger.info(f"Samples with noise:    {num_with_noise}")
        logger.info(f"Clean samples:         {num_clean}")
        logger.info(f"Output directory:      {self.output_dir}")


    def generate_sample(self, idx, audio_subdir):
        selected_voice_name = random.choice(self.voice_names)
        logger.info(f"Sample {idx}: Using randomly selected voice '{selected_voice_name}'")

        error_message = None
        synthesis_successful = False

        noise_file_used = None
        sentence, date_str, time_str = self.generate_sentence()
        speaker_type = random.choice(self.speaker_types)
        noise_type_intended = random.choice(self.noise_types)
        actual_noise_type = "none" # Default if synthesis fails or noise application fails

        clean_audio_dir = os.path.join(audio_subdir, "clean")
        os.makedirs(clean_audio_dir, exist_ok=True)
        base_path = os.path.join(clean_audio_dir, f"clean_{idx + 449}.wav")
        final_path = os.path.join(audio_subdir, f"sample_{idx + 449}.wav")

        # Azure Voice Synthesis
        try:
            ssml_string = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{self.lang}'>
                <voice name='{selected_voice_name}'>
                    {sentence}
                </voice>
            </speak>
            """

            audio_config = speechsdk.audio.AudioOutputConfig(filename=base_path)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)

            logger.info(f"Synthesizing sample {idx} with voice '{selected_voice_name}' to {base_path}...")
            result = synthesizer.speak_ssml_async(ssml_string).get()

            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Successfully synthesized {base_path}")
                synthesis_successful = True # Mark as successful for now
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                error_message = f"Speech synthesis canceled: {cancellation_details.reason}"
                logger.error(f"{error_message} for sample {idx}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    error_message += f" - Details: {cancellation_details.error_details}"
                    logger.error(f"Cancellation Error Details: {cancellation_details.error_details}")
            else:
                error_message = f"Unhandled synthesis result reason: {result.reason}"
                logger.error(f"{error_message} for sample {idx}")

            if synthesis_successful:
                # Check file existence and size
                if not os.path.exists(base_path) or os.path.getsize(base_path) == 0:
                    error_message = f"Azure synthesis reported completed but output file {base_path} is missing or empty."
                    logger.error(f"{error_message} for sample {idx}.")
                    synthesis_successful = False # Mark as failed after check
                else:
                    # File exists and is not empty, proceed with effects/noise
                    try:
                        clean_audio = AudioSegment.from_file(base_path)
                        processed_audio = clean_audio

                        processed_audio = self.apply_voice_effect(processed_audio, speaker_type)

                        if noise_type_intended != "none":
                            noise = self.generate_noise(noise_type_intended, len(processed_audio))
                            if noise and len(noise) >= len(processed_audio):
                                if noise_type_intended == "street" and hasattr(self, "last_noise_file") and self.last_noise_file:
                                    noise_file_used = os.path.basename(self.last_noise_file)
                                else:
                                    noise_file_used = None

                                processed_audio = processed_audio.overlay(noise)
                                processed_audio = effects.normalize(processed_audio)
                                actual_noise_type = noise_type_intended
                                logger.info(f"Applied noise '{actual_noise_type}' to sample {idx}")
                            elif noise is None:
                                logger.warning(f"Noise generation failed for type '{noise_type_intended}' on sample {idx}. Skipping noise.")
                                actual_noise_type = "none"
                            else:
                                logger.warning(f"Generated noise duration mismatch for sample {idx} ({len(noise)}ms vs {len(processed_audio)}ms). Skipping noise.")
                                actual_noise_type = "none"
                        else:
                            actual_noise_type = "none"

                        processed_audio.export(final_path, format="wav")
                        logger.info(f"Exported final sample {idx} to {final_path}")

                    except Exception as e_post:
                        error_message = f"Error during post-processing (noise/export): {e_post}"
                        logger.error(f"{error_message} for sample {idx}", exc_info=True)
                        synthesis_successful = False # Mark as failed if post-processing fails

        except Exception as e_main:
            error_message = f"Unhandled error in generate_sample: {e_main}"
            logger.error(f"{error_message} for idx {idx}", exc_info=True)
            synthesis_successful = False # Mark as failed


        metadata = {
            "id": idx + 449,
            "text": sentence,
            "audio_path": os.path.relpath(final_path, self.output_dir).replace("\\", "/"),
            "clean_audio_path": os.path.relpath(base_path, self.output_dir).replace("\\", "/"),
            "noise_type": actual_noise_type if synthesis_successful else "error",
            "has_noise": actual_noise_type != "none" if synthesis_successful else False,
            "noise_file": noise_file_used if synthesis_successful and actual_noise_type == "street" else None,
            "speaker_type": speaker_type,
            "azure_voice_used": selected_voice_name,
            "date": date_str,
            "time": time_str,
            "generation_status": "success" if synthesis_successful and not error_message else "failed",
            "error": error_message
        }
        return metadata

    def generate_sentence(self):
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
        cars = random.choice(self.cars)
        color = random.choice(['rosso', 'blu', 'verde', 'nero', 'grigio', 'giallo', 'bianco', 'viola', 'arancione', 'argento', 'marrone', 'oro'])
        chassis = f"{random.randint(100000000, 1000000000)}"
        mass = f"{random.randint(1000, 4000)} kg"
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
             "Non è stato possibile contestare per l'assenza del trasgressore. " +
             f"Da decurtare ci sono {random.randint(1, 10)} punti. " +
             (f"Il veicolo è un {cars} di colore {color}, con numero di telaio {chassis} e massa {mass}. " ) +
             f"Necessito la stampa del preavviso tramite {print_method}, in {print_language}, {print_option}. ",
             
             f"Il {date_violation}, alle ore {time_violation}, sto emettendo una notifica di infrazione per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
             f"L'infrazione si è verificata presso {street}, al civico {civico}, violando il codice {violation_type}, in particolare l'articolo {article}, comma {comma} riguardante il {violation}. " +
             f"Devo decurtare {random.randint(1, 10)} punti. " +
             ("Applico una sanzione accessoria in base alle circostanze. " if random.choice([True, False]) else "Non è necessaria alcuna sanzione accessoria. ") +
             (f"Il {vehicle_type} è di colore {color}, con numero di telaio {chassis} e massa {mass}. " ) +
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
             f"Sono previsti {random.randint(1, 10)} punti di decurtazione. " +
             f"Si prega di stampare la presente notifica tramite {print_method}, in lingua {print_language}, {print_option}. ",

             f"Attenzione, sto generando un verbale per il veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per un'infrazione commessa il {date_violation} alle {time_violation}. " +
             f"L'infrazione è avvenuta in {street}, al numero {civico}, dove il veicolo era in {violation}. L'articolo violato è il {article}, comma {comma} del codice {violation_type}. " +
             f"Verranno tolti {random.randint(1, 10)} punti dalla patente. " +
             f"Stampa questo verbale via {print_method} in {print_language}, {print_option}. ",

             f"Procedo con la stesura di un verbale per il veicolo {vehicle_type}, marca {cars}, targato {plate} (di tipo {kind_plate}), proveniente da {nationality}. L'infrazione è stata rilevata il {date_violation} alle ore {time_violation}. " +
             f"Il veicolo si trovava in {street}, al civico {civico}, in {violation}, violando l'articolo {article}, comma {comma}, del codice {violation_type}. " +
             (f"L'agente {agent} ha provveduto alla contestazione immediata dell'infrazione. " if immediate_contestation else "Non è stato possibile effettuare la contestazione immediata a causa dell'assenza del trasgressore. ") +
             f"La violazione comporta la perdita di {random.randint(1, 10)} punti. Richiedo la stampa del verbale in {print_language} tramite {print_method}, {print_option}. ",

             f"Emetto un avviso di violazione, quindi si tratta di un preavviso, per il veicolo {cars}, {vehicle_type}, targato {plate} con targa {kind_plate}, registrato in {nationality}, per un fatto accaduto il {date_violation} alle {time_violation}. " +
             f"Il veicolo si trovava in {street}, al civico {civico}, dove è stato riscontrato {violation}, in base all'articolo {article}, comma {comma}, del codice {violation_type}. " +
             f"Questa infrazione comporta una decurtazione di {random.randint(1, 10)} punti. " +
             f"Genera la stampa dell'avviso via {print_method} in {print_language}, {print_option}. ",

             f"Redazione verbale in corso per il veicolo {vehicle_type} di marca {cars}, con targa {plate} (tipo {kind_plate}), proveniente da {nationality}. Fatto avvenuto il {date_violation} alle ore {time_violation}. " +
             f"Posizione: {street}, numero civico {civico}. Infrazione: {violation}, in violazione dell'articolo {article}, comma {comma}, del codice {violation_type}. " +
             f"Decurtazione prevista: {random.randint(1, 10)} punti. " +
             (f"Si applica contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Impossibile contestare immediatamente per assenza del trasgressore. ") +
             f"Stampa necessaria tramite {print_method} in {print_language}, {print_option}. ",

             f"Sto emettendo un verbale per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
             f"L'infrazione è avvenuta in {street}, civico {civico}: {violation}. Si applica l'articolo {article}, comma {comma} del codice {violation_type}. " +
             f"A seguito della violazione verranno decurtati {random.randint(1, 10)} punti. " +
             (f"Il verbale è stato contestato immediatamente dall'agente {agent}. " if immediate_contestation else "A causa dell'assenza del trasgressore non è stato possibile effettuare la contestazione immediata. ") +
             f"Si prega di procedere con la stampa in {print_language} tramite {print_method}, {print_option}. ",

             f"Si notifica un preavviso relativo al veicolo {vehicle_type} targato {plate} di tipo {kind_plate} proveniente da {nationality}, marca {cars}. " +
             f"La violazione, ovvero {violation}, si è verificata in data {date_violation} alle ore {time_violation} in {street}, al numero {civico}. " +
             f"L'infrazione rientra nell'articolo {article}, comma {comma} del codice {violation_type}. " +
             (f"Contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Non è stato possibile contestare immediatamente a causa dell'assenza del trasgressore. ") +
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
             (f"La contestazione è stata immediata, effettuata dall'agente {agent}. " if immediate_contestation else f"Non è stato possibile effettuare la contestazione immediata, per assenza del trasgressore. ") +
             f"Dalla violazione derivano {random.randint(1, 10)} punti di decurtazione. Stampa necessaria in lingua {print_language} via {print_method}, {print_option}. ",

             f"Emissione di preavviso per il veicolo {vehicle_type}, marca {cars}, targato {plate} di tipo {kind_plate}, proveniente da {nationality}. In data {date_violation} alle ore {time_violation}, è stata rilevata la violazione di {violation} in {street}, al numero {civico}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
             f"Per tale infrazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa del preavviso tramite {print_method} in lingua {print_language}, {print_option}. ",

             f"Si sta redigendo verbale per violazione accertata il {date_violation} alle ore {time_violation} in {street}, al civico {civico}, a carico del veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione contestata è {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
             (f"La contestazione è avvenuta immediatamente da parte dell'agente {agent}. " if immediate_contestation else f"Non è stato possibile contestare immediatamente, per assenza del trasgressore. ") +
             f"Sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa in lingua {print_language} via {print_method}, {print_option}. ",

             f"Si notifica la presenza di un preavviso di violazione per il veicolo {vehicle_type}, marca {cars}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione è avvenuta il {date_violation} alle ore {time_violation} in {street}, al numero civico {civico}, ed è relativa a {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}. " +
             f"Per tale violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa di tale preavviso in lingua {print_language} tramite {print_method}, {print_option}. ",

             f"Sto emettendo un verbale per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per infrazione commessa in data {date_violation}, ore {time_violation}. In {street}, civico {civico}, è stato rilevato {violation}, ai sensi dell'articolo {article}, comma {comma}, del codice {violation_type}. " +
             (f"Contestazione immediata eseguita dall'agente {agent}. " if immediate_contestation else f"Impossibile effettuare la contestazione immediata, assenza del trasgressore. ") +
             f"La violazione comporta la decurtazione di {random.randint(1, 10)} punti. Richiesta stampa in {print_language} via {print_method}, {print_option}. ",

             f"Si notifica un preavviso per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, proveniente da {nationality}. Il giorno {date_violation}, alle ore {time_violation}, in {street}, al numero civico {civico}, è stata commessa l'infrazione di {violation}, come previsto dall'articolo {article}, comma {comma}, del codice {violation_type}. " +
             f"Per questa violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Chiedo la stampa del preavviso in {print_language} via {print_method}, {print_option}. ",
         ]
        selected_sentence = random.choice(templates)
        return selected_sentence, date_violation, time_violation

    def generate_noise(self, noise_type, duration_ms, volume_db=-10.0):
        self.last_noise_file = None
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
                return noise[:duration_ms].apply_gain(volume_db)
            elif noise_type == "none":
                 return None
            else:
                logger.warning(f"Unknown noise type requested: {noise_type}")
                return None
        except Exception as e:
            logger.error(f"Error generating noise type '{noise_type}': {e}", exc_info=True)
            return None

    def apply_voice_effect(self, audio, speaker_type):
         logger.debug(f"Applying effect '{speaker_type}'")
         try:
             if speaker_type == "fast":
                 return audio.speedup(playback_speed=1.2)
             elif speaker_type == "slow":
                 return audio.speedup(playback_speed=0.85)
             elif speaker_type == "radio":
                 filtered_audio = effects.high_pass_filter(audio, cutoff=300)
                 filtered_audio = effects.low_pass_filter(filtered_audio, cutoff=3000)
                 return filtered_audio.apply_gain(-3)
             elif speaker_type == "original":
                  return audio
             else:
                  logger.warning(f"Unknown speaker type effect '{speaker_type}'. Returning original audio.")
                  return audio
         except Exception as e:
              logger.error(f"Error applying voice effect '{speaker_type}': {e}", exc_info=True)
              return audio

    def save_metadata(self, metadata_dir):
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

    def save_sentences(self, sentences, sentence_dir):
        os.makedirs(sentence_dir, exist_ok=True)
        sentences_path = os.path.join(sentence_dir, "sentences.json")
        try:
            with open(sentences_path, "w", encoding="utf-8") as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
            logger.info(f"Sentences saved to: {sentences_path}")
        except IOError as e:
            logger.error(f"Failed to save sentences to {sentences_path}: {e}")


if __name__ == "__main__":
    NUM_SAMPLES = 51
    OUTPUT_BASE = "data/synthetic_datasets/new_data/AzureTTS"
    LANG = "it-IT"

    cognitive_svc_key = os.getenv("COGNITIVE_SERVICE_KEY")
    cognitive_svc_location = os.getenv("COGNITIVE_SERVICE_LOCATION")

    if not cognitive_svc_key or not cognitive_svc_location:
        logger.error("COGNITIVE_SERVICE_KEY or COGNITIVE_SERVICE_LOCATION not found in environment (check .env file). Cannot proceed.")
    else:
        logger.info(f"Cognitive Services credentials loaded for region: {cognitive_svc_location}")

        standard_italian_voices = [
            "it-IT-AlessioMultilingualNeural",
            "it-IT-IsabellaMultilingualNeural",
            "it-IT-GiuseppeMultilingualNeural",
            "it-IT-MarcelloMultilingualNeural",
            "it-IT-GiuseppeNeural",
            "it-IT-ElsaNeural",
            "it-IT-IsabellaNeural",
            "it-IT-DiegoNeural",
            "it-IT-BenignoNeural",
            "it-IT-CalimeroNeural",
            "it-IT-CataldoNeural",
            "it-IT-FabiolaNeural",
            "it-IT-FiammaNeural",
            "it-IT-GianniNeural",
            "it-IT-ImeldaNeural",
            "it-IT-IrmaNeural",
            "it-IT-LisandroNeural",
            "it-IT-PalmiraNeural",
            "it-IT-RinaldoNeural",
        ]

        if not standard_italian_voices:
             logger.error("The list 'standard_italian_voices' is empty. Please add standard Italian voice names.")
        else:
            config_name = "Standard_Italian_Voices"
            output_directory = OUTPUT_BASE
            logger.info(f"\nGenerating dataset for: {config_name}...")
            logger.info(f"Using {len(standard_italian_voices)} standard voices for random selection.")

            try:
                generator = AzureSyntheticDatasetGenerator(
                    output_dir=output_directory,
                    num_samples=NUM_SAMPLES,
                    lang=LANG,
                    voice_names=standard_italian_voices,
                    azure_key=cognitive_svc_key,
                    azure_region=cognitive_svc_location,
                    seed=456734
                )
                generator.generate_all()
                logger.info(f"Finished dataset generation for {config_name}")

            except ValueError as ve:
                 logger.error(f"Configuration error: {ve}")
            except speechsdk.exceptions.ConnectionFailureException as cfe:
                 logger.error(f"Connection Failure (Region: {cognitive_svc_location}): {cfe}. Check region, key, and network.")
            except Exception as ex:
                logger.error(f"Failed to generate dataset: {ex}", exc_info=True)

    logger.info("\n--- Dataset Generation Task Complete ---")