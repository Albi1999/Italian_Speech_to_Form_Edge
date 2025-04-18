import os
import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm

from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine

from gtts import gTTS

class SyntheticDatasetGenerator:
    def __init__(
        self,
        output_dir="data/synthetic_datasets/GoogleTTS",
        num_samples=100,
        lang="it",
        seed=42
    ):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.lang = lang
        self.seed = seed
        random.seed(seed)
        os.makedirs(self.output_dir, exist_ok=True)
        self.samples = []

        # Voice effects and noise types
        self.noise_types = ["none", "white", "static", "hum", "street"]
        self.speaker_types = ["male", "female", "radio", "fast", "slow"]
        self.street_noise_files = [
            os.path.join("data", "noise", "ambiance_traffic_people_on_the_street.wav"),
            os.path.join("data", "noise", "street-from-courtyard.mp3"),
            os.path.join("data", "noise", "street-raining.wav")
        ]

        # Static data for generating sentences
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

        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            try:
                sample = self.generate_sample(i)
                if sample:
                    self.samples.append(sample)
                    if sample["noise_type"] != "none":
                        num_with_noise += 1
                    else:
                        num_clean += 1
                    sentences.append(sample["text"])
            except Exception as e:
                print(f"Error in generating sample {i}: {e}")

        self.save_metadata()
        self.save_sentences(sentences)

        print("\n Noise statistics:")
        print(f"\n Samples with noise: {num_with_noise}")
        print(f"\n Clean samples:      {num_clean}")


    def generate_sample(self, idx):
        noise_file_used = None
        sentence, date_str, time_str = self.generate_sentence()
        speaker_type = random.choice(self.speaker_types)
        noise_type = random.choice(self.noise_types)

        audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        clean_dir = os.path.join(audio_dir, "clean")
        os.makedirs(clean_dir, exist_ok=True)
        base_path = os.path.join(clean_dir, f"clean_{idx}.wav")
        final_path = os.path.join(audio_dir, f"sample_{idx}.wav")

        # Voice synthesis
        tts = gTTS(sentence, lang=self.lang)
        tts.save(base_path)

        clean = AudioSegment.from_file(base_path)
        # Make a copy of the original clean before applying voice/noise
        original_clean = clean

        # Check TTS quality
        if len(clean) < 30000 or clean.dBFS < -40:
            raise ValueError("\n Audio too short or silent")

        # Apply voice effect
        clean = self.apply_voice_effect(clean, speaker_type)

        try:
            if noise_type != "none":
                noise = self.generate_noise(noise_type, len(clean))
                if noise and len(noise) >= len(clean):
                    noise_file_used = os.path.basename(self.last_noise_file) if hasattr(self, "last_noise_file") else None
                    clean = clean.overlay(noise)
                    clean = effects.normalize(clean)
                else:
                    print(f"\n Skipped noise overlay for sample {idx} (invalid noise: {noise_type})")

            clean.export(final_path, format="wav")

            # Validate final audio
            if len(clean) < 30000:
                raise ValueError(f"\n Corrupted output (too short): {final_path}")

        except Exception as e:
            print(f"\n Using clean audio for sample {idx} due to overlay failure: {e}")
            # Export the original clean (before noise/effects)
            original_clean.export(final_path, format="wav")

        return {
            "id": idx,
            "text": sentence,
            "audio_path": os.path.basename(final_path),
            "noise_type": noise_type,
            "has_noise": noise_type != "none",
            "noise_file": noise_file_used,
            "speaker_type": speaker_type,
            "date": date_str,
            "time": time_str
        }

    def regenerate_samples(self, sample_indices):
        """ Regenerate specific samples identified by their indices. """
        regenerated_samples = []

        # Load existing metadata
        metadata_file_path = os.path.join(self.output_dir, "metadata", "samples.json")
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                existing_samples = json.load(f)
        else:
            existing_samples = []

        for idx in sample_indices:
            try:
                sample = self.generate_sample(idx)
                if sample:
                    regenerated_samples.append(sample)
            except Exception as e:
                print(f"Error in regenerating sample {idx}: {e}")

        # Update the metadata list
        id_to_sample = {sample['id']: sample for sample in existing_samples}
        for sample in regenerated_samples:
            id_to_sample[sample['id']] = sample

        # Convert back to list
        updated_samples = list(id_to_sample.values())

        # Save updated metadata
        metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        with open(os.path.join(metadata_dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump(updated_samples, f, indent=2, ensure_ascii=False)

        print("\nRegeneration completed for samples:", sample_indices)

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
            f"Il giorno {date_violation}, alle ore {time_violation}, è stato redatto un verbale per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}." +
            f"L'infrazione è avvenuta in {street}, al civico {civico}. È stata riscontrata una violazione del codice {violation_type}, articolo {article}, comma {comma} ovvero {violation}" +
            (f"Il verbale è stato eseguito sul posto, con contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Non è stato possibile contestare sul posto a causa dell'assenza del trasgressore. ") +
            f"Sono previsti {random.randint(1, 10)} punti da decurtare, e non è stata applicata alcuna sanzione accessoria. Stampa in {print_language} tramite {print_method}, {print_option}.",

            f"In data {date_violation}, alle ore {time_violation}, è stato notificato un preavviso per un {vehicle_type} targato {plate} con targa {kind_plate}, immatricolato in {nationality}. " +
            f"L'infrazione è stata rilevata in {street}, al civico {civico}, con violazione del codice {violation_type}, articolo {article}, comma {comma}, in particolare si è riscontrato un {violation} " +
            "Non è stato possibile contestare per l'assenza del trasgressore, e per questo è indicato il motivo di mancata contestazione. " +
            f"Da decurtare ci sono {random.randint(1, 10)} punti. " +
            (f"Il veicolo è un {cars} di colore {color}, con numero di telaio {chassis} e massa {mass}." if random.choice([True, False]) else "Ulteriori dettagli sul veicolo non sono necessari. ") +
            f"Necessito la stampa del preavviso tramite {print_method}, in {print_language}, {print_option}.",

            f"Il {date_violation}, alle ore {time_violation}, sto emettendo una notifica di infrazione per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
            f"L'infrazione si è verificata presso {street}, al civico {civico}, violando il codice {violation_type}, in particolare l'articolo {article}, comma {comma} riguardante il {violation}." +
            f"Devo decurtare {random.randint(1, 10)} punti. " +
            ("Applico una sanzione accessoria in base alle circostanze. " if random.choice([True, False]) else "Non è necessaria alcuna sanzione accessoria. ") +
            (f"Il {vehicle_type} è di colore {color}, con numero di telaio {chassis} e massa {mass}." if random.choice([True, False]) else f"Non sono necessari ulteriori dettagli sul {vehicle_type}. ") +
            f"Stampamelo con il {print_method} in {print_language}, {print_option}.",

            f"Alle {time_violation} del {date_violation}, sto redigendo un verbale per il veicolo {vehicle_type} targato {plate} con targa {kind_plate}, immatricolato in {nationality}" +
            f"L'infrazione è stata accertata in {street}, al civico {civico}, con violazione del codice {violation_type}, articolo {article},comma {comma} per {violation} " +
            f"Sono previsti {random.randint(1, 10)} punti da decurtare. " +
            (f"Contestazione immediata effettuata dall'agente {agent}, il verbale è stato redatto sul posto in {street}. " if immediate_contestation else "Non è stata possibile una contestazione immediata per l'assenza del trasgressore. ") +
            f"Non posso applicare alcuna sanzione accessoria. Stampa tramite {print_method}, in {print_language}, {print_option}.",

            f"Verbale redatto in data {date_violation} alle ore {time_violation} per la violazione riscontrata al veicolo {vehicle_type}, marca {cars}, targato {plate} (tipo {kind_plate}) proveniente da {nationality}. " +
            f"In particolare, in {street}, civico {civico}, è stato accertato {violation}, in violazione dell'articolo {article}, comma {comma} del codice {violation_type}. " +
            (f"L'agente {agent} ha proceduto con contestazione immediata. " if immediate_contestation else "Data l'assenza del conducente, la contestazione non è stata immediata.") +
            f"Si prevede la decurtazione di {random.randint(1, 10)} punti dalla patente. Richiedo la stampa in {print_language} via {print_method}, {print_option}.",

            f"Si notifica che in data {date_violation}, alle ore {time_violation}, è stato emesso un preavviso di violazione per il veicolo {vehicle_type} di marca {cars}, con targa {plate} (tipo {kind_plate}), registrato in {nationality}. " +
            f"L'infrazione è avvenuta in {street}, al numero {civico}, e consiste in {violation}, come previsto dall'articolo {article}, comma {comma} del codice {violation_type}. " +
            "Non essendo stato possibile identificare il conducente sul posto, la notifica viene inviata al proprietario del veicolo. " +
            f"Sono previsti {random.randint(1, 10)} punti di decurtazione.  " +
            f"Si prega di stampare la presente notifica tramite {print_method}, in lingua {print_language}, {print_option}.",

            f"Attenzione, sto generando un verbale per il veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per un'infrazione commessa il {date_violation} alle {time_violation}. " +
            f"L'infrazione è avvenuta in {street}, al numero {civico}, dove il veicolo era in {violation}.  L'articolo violato è il {article}, comma {comma} del codice {violation_type}. " +
            f"Verranno tolti {random.randint(1, 10)} punti dalla patente. " +
            f"Stampa questo verbale via {print_method} in {print_language}, {print_option}.",

            f"Procedo con la stesura di un verbale per il veicolo {vehicle_type}, marca {cars}, targato {plate} (di tipo {kind_plate}), proveniente da {nationality}.  L'infrazione è stata rilevata il {date_violation} alle ore {time_violation}. " +
            f"Il veicolo si trovava in {street}, al civico {civico}, in {violation}, violando l'articolo {article}, comma {comma}, del codice {violation_type}. " +
            (f"L'agente {agent} ha provveduto alla contestazione immediata dell'infrazione. " if immediate_contestation else "Non è stato possibile effettuare la contestazione immediata a causa dell'assenza del trasgressore, quindi motivo mancata contestazione: assenza del trasgressore. ") +
            f"La violazione comporta la perdita di {random.randint(1, 10)} punti.  Richiedo la stampa del verbale in {print_language} tramite {print_method}, {print_option}.",

            f"Emetto un avviso di violazione, quindi si tratta di un preavviso, per il veicolo {cars}, {vehicle_type}, targato {plate} con targa {kind_plate}, registrato in {nationality},  per un fatto accaduto il {date_violation} alle {time_violation}. " +
            f"Il veicolo si trovava in {street}, al civico {civico}, dove è stato riscontrato {violation}, in base all'articolo {article}, comma {comma}, del codice {violation_type}. " +
            f"Questa infrazione comporta una decurtazione di {random.randint(1, 10)} punti.  " +
            f"Genera la stampa dell'avviso via {print_method} in {print_language}, {print_option}.",

            f"Redazione verbale in corso per il veicolo {vehicle_type} di marca {cars}, con targa {plate} (tipo {kind_plate}), proveniente da {nationality}.  Fatto avvenuto il {date_violation} alle ore {time_violation}. " +
            f"Posizione: {street}, numero civico {civico}. Infrazione: {violation}, in violazione dell'articolo {article}, comma {comma}, del codice {violation_type}. " +
            f"Decurtazione prevista: {random.randint(1, 10)} punti. " +
            (f"Si applica contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Impossibile contestare immediatamente per assenza del trasgressore, motivo: assenza del trasgressore. ") +
            f"Stampa necessaria tramite {print_method} in {print_language}, {print_option}.",

            f"Sto emettendo un verbale per il veicolo {cars}, {vehicle_type} immatricolato in {nationality} con targa {plate} di tipo {kind_plate}. " +
            f"L'infrazione è avvenuta in {street}, civico {civico}: {violation}. Si applica l'articolo {article}, comma {comma} del codice {violation_type}. " +
            f"A seguito della violazione verranno decurtati {random.randint(1, 10)} punti." +
            (f"Il verbale è stato contestato immediatamente dall'agente {agent}. " if immediate_contestation else "A causa dell'assenza del trasgressore non è stato possibile effettuare la contestazione immediata. ") +
            f"Si prega di procedere con la stampa in {print_language} tramite {print_method}, {print_option}.",

            f"Si notifica un preavviso relativo al veicolo {vehicle_type} targato {plate} di tipo {kind_plate} proveniente da {nationality}, marca {cars}. " +
            f"La violazione, ovvero {violation}, si è verificata in data {date_violation} alle ore {time_violation} in {street}, al numero {civico}. " +
            f"L'infrazione rientra nell'articolo {article}, comma {comma} del codice {violation_type}. " +
            (f"Contestazione immediata da parte dell'agente {agent}. " if immediate_contestation else "Non è stato possibile contestare immediatamente a causa dell'assenza del trasgressore, motivo mancata contestazione: assenza del trasgressore. ") +
            f"La sanzione prevede la perdita di {random.randint(1, 10)} punti. Si richiede la stampa in lingua {print_language}, tramite {print_method}, {print_option}.",

            f"Sto procedendo con la redazione di un verbale per {vehicle_type}, marca {cars}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. " +
            f"In data {date_violation} alle ore {time_violation}, in {street}, al civico {civico}, è stata commessa la seguente violazione: {violation} (articolo {article}, comma {comma} del codice {violation_type})." +
            f"Dalla violazione consegue la perdita di {random.randint(1, 10)} punti." +
            f"Stampa in modalità {print_method} e in lingua {print_language}, {print_option}.",

            f"Si avvisa che è in corso la stesura di un preavviso di violazione per il veicolo {vehicle_type} con targa {plate} di tipo {kind_plate}, marca {cars}, proveniente da {nationality}. " +
            f"L'infrazione, nello specifico {violation}, si è verificata in data {date_violation} alle ore {time_violation} in {street}, al numero {civico} (articolo {article}, comma {comma} del codice {violation_type}). " +
            f"Si comunica che dalla violazione conseguirà una decurtazione di {random.randint(1, 10)} punti." +
            f"Si richiede la stampa in lingua {print_language}, modalità {print_method}, {print_option}.",

            f"In data {date_violation}, alle ore {time_violation}, si eleva verbale al veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per violazione dell'articolo {article}, comma {comma} del codice {violation_type}, ovvero {violation}, rilevata in {street}, al civico {civico}." +
            (f"La contestazione è stata immediata, effettuata dall'agente {agent}. " if immediate_contestation else f"Non è stato possibile effettuare la contestazione immediata, motivo: assenza del trasgressore. ") +
            f"Dalla violazione derivano {random.randint(1, 10)} punti di decurtazione. Stampa necessaria in lingua {print_language} via {print_method}, {print_option}.",

            f"Emissione di preavviso per il veicolo {vehicle_type}, marca {cars}, targato {plate} di tipo {kind_plate}, proveniente da {nationality}. In data {date_violation} alle ore {time_violation}, è stata rilevata la violazione di {violation} in {street}, al numero {civico}, in base all'articolo {article}, comma {comma} del codice {violation_type}." +
            f"Per tale infrazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa del preavviso tramite {print_method} in lingua {print_language}, {print_option}.",

            f"Si sta redigendo verbale per violazione accertata il {date_violation} alle ore {time_violation} in {street}, al civico {civico}, a carico del veicolo {cars}, {vehicle_type} con targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione contestata è {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}." +
            (f"La contestazione è avvenuta immediatamente da parte dell'agente {agent}. " if immediate_contestation else f"Non è stato possibile contestare immediatamente, motivo: assenza del trasgressore. ") +
            f"Sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa in lingua {print_language} via {print_method}, {print_option}.",

            f"Si notifica la presenza di un preavviso di violazione per il veicolo {vehicle_type}, marca {cars}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}. La violazione è avvenuta il {date_violation} alle ore {time_violation} in {street}, al numero civico {civico}, ed è relativa a {violation}, in base all'articolo {article}, comma {comma} del codice {violation_type}." +
            f"Per tale violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Si richiede la stampa di tale preavviso in lingua {print_language} tramite {print_method}, {print_option}.",

            f"Sto emettendo un verbale per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, immatricolato in {nationality}, per infrazione commessa in data {date_violation}, ore {time_violation}. In {street}, civico {civico}, è stato rilevato {violation}, ai sensi dell'articolo {article}, comma {comma}, del codice {violation_type}." +
            (f"Contestazione immediata eseguita dall'agente {agent}. " if immediate_contestation else f"Impossibile effettuare la contestazione immediata, motivo: assenza del trasgressore. ") +
            f"La violazione comporta la decurtazione di {random.randint(1, 10)} punti. Richiesta stampa in {print_language} via {print_method}, {print_option}.",

            f"Si notifica un preavviso per il veicolo {cars} {vehicle_type}, targa {plate} di tipo {kind_plate}, proveniente da {nationality}. Il giorno {date_violation}, alle ore {time_violation}, in {street}, al numero civico {civico}, è stata commessa l'infrazione di {violation}, come previsto dall'articolo {article}, comma {comma}, del codice {violation_type}." +
            f"Per questa violazione sono previsti {random.randint(1, 10)} punti di decurtazione. Chiedo la stampa del preavviso in {print_language} via {print_method}, {print_option}.",
        ]

        selected_sentence = random.choice(templates)
        print(f"Generated sentence: {selected_sentence}")
        return selected_sentence, date_violation, time_violation


    def generate_noise(self, noise_type, duration_ms, volume_db=-10.0):
        if noise_type == "white":
            return WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(volume_db)

        elif noise_type == "static":
            freqs = [1000, 1200, 1500]
            waves = [Sine(f).to_audio_segment(duration=duration_ms).apply_gain(volume_db - 5) for f in freqs]
            return sum(waves)

        elif noise_type == "hum":
            return Sine(60).to_audio_segment(duration=duration_ms).apply_gain(volume_db)

        elif noise_type == "street":
            noise_path = random.choice(self.street_noise_files)
            self.last_noise_file = noise_path
            try:
                if not os.path.exists(noise_path):
                    print(f"\n Noisy file missing: {noise_path}")
                    return None

                noise = AudioSegment.from_file(noise_path)

                noise = noise.set_channels(1).set_frame_rate(22050)

                if len(noise) < duration_ms:
                    noise *= (duration_ms // len(noise) + 1)

                return noise[:duration_ms].apply_gain(volume_db)

            except Exception as e:
                print(f"\n Error: failed to load noise '{os.path.basename(noise_path)}': {e}")
                return None

        return None



    def apply_voice_effect(self, audio, speaker_type):
        if speaker_type == "fast":
            return audio.speedup(playback_speed=1.2)
        elif speaker_type == "slow":
            return audio.speedup(playback_speed=0.85)
        elif speaker_type == "radio":
            return effects.low_pass_filter(audio, cutoff=3000).apply_gain(-3)
        elif speaker_type == "male":
            return audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * 0.95)}).set_frame_rate(audio.frame_rate)
        elif speaker_type == "female":
            return audio.speedup(playback_speed=1.05)
        return audio

    def save_metadata(self):
        metadata_dir = os.path.join(self.output_dir, "meta")
        os.makedirs(metadata_dir, exist_ok=True)

        with open(os.path.join(metadata_dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)

        print("Metadata saved in:", metadata_dir)
    
    def save_sentences(self, sentences):
        sentence_dir = os.path.join(self.output_dir, "tx")
        os.makedirs(sentence_dir, exist_ok=True)
        sentences_path = os.path.join(sentence_dir, "sentences.json")
        try:
            with open(sentences_path, "w", encoding="utf-8") as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)
            print("Sentences saved in:", sentence_dir)
        except IOError as e:
            print(f"Failed to save sentences to {sentences_path}: {e}")

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator(num_samples=100)
    generator.generate_all()

    # Regenerate specific corrupted samples
    #corrupted_samples = [27, 34, 65, 72, 80, 83, 85, 88]
    #generator.regenerate_samples(corrupted_samples)
