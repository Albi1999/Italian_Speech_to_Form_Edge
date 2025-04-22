# Form

1. Verbale o preavviso
2. Data violazione
3. Ora violazione
4. Data verbalizzazione (solo se la multa è contestata seduta stante)
5. Ora verbalizzazione (solo se la multa è contestata seduta stante)
6. Luogo della verbalizzazione
7. Indirizzo (della violazione):
    1. Strada 1 (via)
    2. Civico 1 (civico)
    3. Strada 2 (se necessario)
    4. Civico 2 (se necessario)
    5. Civico esteso (?)
8. Veicolo:
    1. Tipologia (autovettura, camion ecc..)
    2. Nazione (del veicolo, si può prendere dalla targa, codice ISO alpha 3)
    3. Targa
    4. Tipologia targa (ufficiale, speciale, prova ecc)
    5. Marca/Modello
    6. Colore
    7. Telaio
    8. Massa
    9. Note veicolo
9. Contestazione:
    1. Contestazione immediata (bool)
        IF True AND Trasgressore acconsente
        1. Anagrafiche
            1. Anagrafica
                1. Persona fisica
                    1. Denominazione
                    2. Codice fiscale
                2. Persona giuridica
                    1. (?)
                    2. …
            2. Targa
                1. targa(?)
                2. tipologia targa(?)
            3. Documento
                1. Tipo documento (?)
                2. Numero documento (?)
    2. Motivo mancata contestazione (selezione da un elenco definito)
        (solitamente in questo caso viene utilizzato: Assenza del trasgressore e dell’obbligato in solido)
    3. Descrizione mancata contestazione (penso ripeta il motivo della mancata contestazione o almeno nell’esempio che ci hanno mostrato è così)
10. Violazioni (gli articoli vendono presi dal database EVO)
    1. Codice
    2. Articolo
    3. Comma
    4. Sanzione accessoria (bool)
        l’agente può decidere se applicare o meno la sanzione accessoria in base alla possibilità/necessità di applicarla nel specifico caso (disponibilità o meno di carri attrezzi liberi al momento in un caso di rimozione forzata come sanzione accessoria)
11. Punti (int - inserire numero di punti decurtati o da decurtare)
12. Motivazioni (in automatico la descrizione dell’articolo violato oppure una di quelle già impostate su concilia evo)
13. Dichiarazioni (del trasgressore se presente)
14. Annotazioni (eventuali annotazioni dell’agente)
15. Tipo stampa (bool (Bluetooth - WIFI))
16. Lingua stampa (Italiano (Non mi sembra di aver visto ulteriori opzioni))
17. Stampa anche comunicazione (bool)
