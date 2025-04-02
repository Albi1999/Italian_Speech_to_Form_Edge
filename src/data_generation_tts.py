import os
import json
import random
from datetime import datetime, timedelta
from gtts import gTTS
from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine
from tqdm import tqdm

class SyntheticDatasetGenerator:
    def __init__(
        self,
        output_dir="data/synthetic_dataset",
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
        self.cars = ["Fiat Panda", "Volkswagen Golf", "Renault Clio", "BMW Serie 1", "Opel Corsa", "Toyota Yaris", "Audi A3", "Mercedes Classe A"]
        self.plates = ["AB123CD", "CD456EF", "GH789IJ", "KL321MN", "XY987ZT", "ZA456QP", "FG234LM", "TR098YU"]
        self.agents = ["Mario Rossi", "Anna Bianchi", "Carlo Bruni", "Silvia Neri", "Luca Verdi", "Giulia Costa", "Marco Gentili", "Sara Bellini"]
        self.streets = [
            "via del Corso 12, Roma", "via Milano 24, Torino", "via Napoli 5, Napoli",
            "piazza Venezia, Roma", "via Garibaldi 11, Firenze", "corso Buenos Aires 98, Milano",
            "via Etnea 45, Catania", "via Roma 77, Palermo"
        ]
        self.violations = [
            "divieto di sosta", "doppia fila", "parcheggio su strisce pedonali",
            "accesso vietato", "zona rimozione", "sosta davanti a passo carrabile"
        ]
        self.articles = [
            "articolo 158, comma 2", "articolo 7, comma 15", "articolo 157, comma 5"
        ]

    def generate_all(self):
        num_with_noise = 0
        num_clean = 0

        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            try:
                sample = self.generate_sample(i)
                if sample:
                    self.samples.append(sample)
                    if sample["noise_type"] != "none":
                        num_with_noise += 1
                    else:
                        num_clean += 1
            except Exception as e:
                print(f"Error in generating sample {i}: {e}")

        self.save_metadata()

        print("\n Noise statistics:")
        print(f"\n Samples with noise: {num_with_noise}")
        print(f"\n Clean samples:      {num_clean}")


    def generate_sample(self, idx):
        noise_file_used = None
        sentence, date_str, time_str = self.generate_sentence()
        speaker_type = random.choice(self.speaker_types)
        noise_type = random.choice(self.noise_types)

        base_path = os.path.join(self.output_dir, f"clean_{idx}.wav")
        final_path = os.path.join(self.output_dir, f"sample_{idx}.wav")

        # Voice synthesis
        tts = gTTS(sentence, lang=self.lang)
        tts.save(base_path)

        clean = AudioSegment.from_file(base_path)
        # Make a copy of the original clean before applying voice/noise
        original_clean = clean

        # Check TTS quality
        if len(clean) < 5000 or clean.dBFS < -40:
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
            if len(clean) < 5000:
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

    def generate_sentence(self):
        agent = random.choice(self.agents)
        car = random.choice(self.cars)
        plate = random.choice(self.plates)
        street = random.choice(self.streets)
        violation = random.choice(self.violations)
        article = random.choice(self.articles)
        dt = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        date_str = dt.strftime("%d/%m/%Y")
        time_str = dt.strftime("%H:%M")

        templates = [
            f"Sono l'agente {agent}. Alle ore {time_str} del {date_str}, ho rilevato un'infrazione per {violation} in {street}. Veicolo: {car}, targa {plate}.",
            f"Agente {agent} in servizio. Segnalazione di {violation} alle {time_str} del {date_str}, in {street}. Auto coinvolta: {car}, targa {plate}.",
            f"{date_str}, ore {time_str}: veicolo {car}, targa {plate}, in {violation} presso {street}. Rilevato da agente {agent}.",
            f"Infrazione riscontrata: {violation}. Posizione: {street}. Ora: {time_str}, Data: {date_str}. Auto: {car}, targa {plate}. Agente: {agent}.",
            f"Durante il pattugliamento in zona {street}, l'agente {agent} ha rilevato una violazione del {article}. L'auto coinvolta è una {car} targata {plate}, alle ore {time_str} del {date_str}.",
            f"Il giorno {date_str}, alle ore {time_str}, è stata segnalata un’infrazione per {violation} in {street}. Veicolo coinvolto: {car}, targa {plate}. Agente: {agent}.",
            f"Verbale di infrazione redatto dall'agente {agent}. Data: {date_str}, ora: {time_str}. Infrazione: {violation}. Luogo: {street}. Veicolo: {car}, targa {plate}.",
            f"In base al {article}, l'agente {agent} ha riscontrato un'infrazione in {street} alle ore {time_str} del {date_str}. Veicolo coinvolto: {car}, targa {plate}."
        ]
        return random.choice(templates), date_str, time_str

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
        metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        with open(os.path.join(metadata_dir, "samples.json"), "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)

        print("Metadata saved in:", metadata_dir)

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator(num_samples=100)
    generator.generate_all()
