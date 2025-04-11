import json
from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError
from typing import Optional, List, Literal
from datetime import date, time

# Definizione delle classi
class AnagraficaPersonaFisica(BaseModel):
    denominazione: str
    codice_fiscale: str

class Documento(BaseModel):
    tipo_documento: str
    numero_documento: str

class Anagrafiche(BaseModel):
    persona_fisica: Optional[AnagraficaPersonaFisica] = None
    persona_giuridica: Optional[dict] = None
    documento: Optional[Documento] = None

class Veicolo(BaseModel):
    tipologia: Literal["autovettura", "camion", "ciclomotore", "motoveicolo", "rimorchio", "macchina agricola", "macchina operatrice"]
    nazione: str
    targa: str
    tipologia_targa: Literal["ufficiale", "speciale", "prova", "copertura"]
    marca_modello: str
    colore: str
    telaio: str
    massa: Optional[str] = None
    note_veicolo: Optional[str] = None

class Violazioni(BaseModel):
    codice: str
    articolo: str
    comma: str
    sanzione_accessoria: bool

class TrafficViolationForm(BaseModel):
    verbale_preavviso: Literal["verbale", "preavviso"]
    data_violazione: date
    ora_violazione: time
    data_verbalizzazione: Optional[date] = None
    ora_verbalizzazione: Optional[time] = None
    luogo_verbalizzazione: str
    strada_1: str
    civico_1: str
    strada_2: Optional[str] = None
    civico_2: Optional[str] = None
    civico_esteso: Optional[str] = None
    veicolo: Veicolo
    contestazione_immediata: bool
    motivo_mancata_contestazione: Optional[str] = None
    descrizione_mancata_contestazione: Optional[str] = None
    violazioni: List[Violazioni]
    punti: int
    motivazioni: Optional[str] = None
    dichiarazioni: Optional[str] = None
    annotazioni: Optional[str] = None
    tipo_stampa: Literal["bluetooth", "wifi"]
    lingua_stampa: Literal["italiano"]
    stampa_anche_comunicazione: bool
    trasgressore_acconsente: Optional[bool] = None
    anagrafiche: Optional[Anagrafiche] = None

    @model_validator(mode="before")
    def validate_contestation(cls, values):
        cnt_immediata = values.get('contestazione_immediata')
        if cnt_immediata:
            if 'trasgressore_acconsente' not in values or not values['trasgressore_acconsente']:
                raise ValueError("Trasgressore deve acconsentire se la contestazione è immediata.")
            if 'anagrafiche' not in values:
                raise ValueError("Anagrafiche devono essere fornita se la contestazione è immediata.")
        return values

    @field_validator("descrizione_mancata_contestazione")
    def copy_descrizione_from_motivo(cls, v, values):
        motivo = values.get("motivo_mancata_contestazione")
        if not v and motivo:
            return motivo
        return v

# Esempio di utilizzo del modello
example_data = {
    "verbale_preavviso": "verbale",
    "data_violazione": date(2023, 11, 18),
    "ora_violazione": time(14, 30),
    "luogo_verbalizzazione": "piazza maggiore, bologna",
    "strada_1": "via roma",
    "civico_1": "10",
    "veicolo": {
        "tipologia": "autovettura",
        "nazione": "ITA",
        "targa": "AB123CD",
        "tipologia_targa": "ufficiale",
        "marca_modello": "fiat panda",
        "colore": "rosso",
        "telaio": "12345678901234567"
    },
    "contestazione_immediata": False,
    "motivo_mancata_contestazione": "assenza del trasgressore",
    "violazioni": [
        {
            "codice": "stradale",
            "articolo": "6",
            "comma": "1a(1-12)",
            "sanzione_accessoria": True
        }
    ],
    "punti": 5,
    "dichiarazioni": "Ho rispettato il codice della strada",
    "tipo_stampa": "bluetooth",
    "lingua_stampa": "italiano",
    "stampa_anche_comunicazione": False
}

try:
    traffic_violation_form = TrafficViolationForm(**example_data)
    schema = traffic_violation_form.model_json_schema()
    print(schema)

    # Salva il schema JSON in un file
    with open("traffic_violation_form_schema.json", "w") as f:
        f.write(json.dumps(schema, indent=2))
except ValidationError as e:
    print(f"Errore nella validazione: {e}")