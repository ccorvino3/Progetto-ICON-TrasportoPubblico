# ontology.py
from owlready2 import *
import pandas as pd
from utils.manage_file import get_absolute_path

# Funzione per caricare l'ontologia da file
def get_onto(file_onto):
    file_onto_path = get_absolute_path(file_onto)
    onto = get_ontology(file_onto_path).load()
    return onto

# Funzione per caricare i dati dei treni e popolare l'ontologia
def load_train_data_and_populate_ontology(dataset):
    onto = get_onto("onto.owl")
    with onto:
        for index, row in dataset.iterrows():
            # Creazione delle istanze
            stazione_partenza = onto.search_one(iri="*Stazione", label=row['Departure station'].replace(" ", "_"))
            if not stazione_partenza:
                stazione_partenza = onto.Stazione  # Creazione di una nuova istanza di Stazione in caso di mancato ritrovamento
                stazione_partenza.label = [row['Departure station'].replace(" ", "_")]
            stazione_arrivo = onto.search_one(iri="*Stazione", label=row['Arrival station'].replace(" ", "_"))
            if not stazione_arrivo:
                stazione_arrivo = onto.Stazione  # Creazione di una nuova istanza di Stazione in caso di mancato ritrovamento
                stazione_arrivo.label = [row['Arrival station'].replace(" ", "_")]
                
            treno = onto.Treno(f"Treno_{index}")

            # Collegamento delle proprietà
            treno.haPartenza.append(stazione_partenza)
            treno.haArrivo.append(stazione_arrivo)

            ritardo = onto.Ritardo()
            treno.haRitardo = [ritardo]

            # Popolamento delle proprietà funzionali
            treno.tempoViaggioMedio = [float(row['Average travel time (min)'])]
            treno.numeroCircolazioniPreviste = [int(row['Number of expected circulations'])]
            treno.numeroTreniCancellati = [int(row['Number of cancelled trains'])]
            treno.numeroTreniRitardoPartenza = [int(row['Number of late trains at departure'])]
            treno.ritardoMedioTreniPartenza = [float(row['Average delay of late departing trains (min)'])]
            treno.ritardoMedioTotaleTreniPartenza = [float(row['Average delay of all departing trains (min)'])]
            treno.numeroTreniRitardoArrivo = [int(row['Number of trains late on arrival'])]
            treno.ritardoMedioTreniArrivo = [float(row['Average delay of late arriving trains (min)'])]
            treno.ritardoMedioTotaleTreniArrivo = [float(row['Average delay of all arriving trains (min)'])]
            treno.percentualeRitardiCausaEsterno = [float(row['Delay due to external causes'])]
            treno.percentualeRitardiCausaInfrastruttura = [float(row['Delay due to railway infrastructure'])]
            treno.percentualeRitardiCausaGestioneTraffico = [float(row['Delay due to traffic management'])]
            treno.percentualeRitardiCausaMaterialeRotabile = [float(row['Delay due to rolling stock'])]
            treno.percentualeRitardiCausaGestioneStazione = [float(row['Delay due to station management and reuse of material'])]
            treno.percentualeRitardiCausaPasseggeri = [float(row['Delay due to travellers taken into account'])]
            treno.numeroTreniRitardo15Minuti = [int(row['Number of late trains > 15min'])]
            treno.ritardoMedioTreni15Minuti = [float(row['Average train delay > 15min'])]
            treno.numeroTreniRitardo30Minuti = [int(row['Number of late trains > 30min'])]
            treno.numeroTreniRitardo60Minuti = [int(row['Number of late trains > 60min'])]
            treno.periodo = [row['Period']]

    # Salva l'ontologia aggiornata su file
    onto.save(file="updated_onto.owl", format="rdfxml")
    return onto

# Funzione per analizzare i ritardi
def analyze_delays(onto):
    delay_causes = {
        "CausaEsterno": 0,
        "CausaInfrastruttura": 0,
        "CausaGestioneTraffico": 0,
        "CausaMaterialeRotabile": 0,
        "CausaGestioneStazione": 0,
        "CausaPasseggeri": 0
    }

    for treno in onto.Treno.instances():
        for ritardo in treno.haRitardo:
            # Aggiungi il controllo delle proprietà corrette
            if hasattr(ritardo, 'causatoDaCausaEsterno') and ritardo.causatoDaCausaEsterno:
                delay_causes["CausaEsterno"] += 1
            if hasattr(ritardo, 'causatoDaCausaInfrastruttura') and ritardo.causatoDaCausaInfrastruttura:
                delay_causes["CausaInfrastruttura"] += 1
            if hasattr(ritardo, 'causatoDaCausaGestioneTraffico') and ritardo.causatoDaCausaGestioneTraffico:
                delay_causes["CausaGestioneTraffico"] += 1
            if hasattr(ritardo, 'causatoDaCausaMaterialeRotabile') and ritardo.causatoDaCausaMaterialeRotabile:
                delay_causes["CausaMaterialeRotabile"] += 1
            if hasattr(ritardo, 'causatoDaCausaGestioneStazione') and ritardo.causatoDaCausaGestioneStazione:
                delay_causes["CausaGestioneStazione"] += 1
            if hasattr(ritardo, 'causatoDaCausaPasseggeri') and ritardo.causatoDaCausaPasseggeri:
                delay_causes["CausaPasseggeri"] += 1

    return delay_causes

# Funzione per generare il rapporto
def generate_report(delay_causes):
    report = "Rapporto delle cause dei ritardi:\n"
    total_delays = sum(delay_causes.values())
    for cause, count in delay_causes.items():
        percentage = (count / total_delays) * 100 if total_delays > 0 else 0
        report += f"{cause}: {count} ritardi ({percentage:.2f}%)\n"
    
    print(report)

# Funzione per eseguire l'ontologia e generare il rapporto delle cause dei ritardi dei treni
def onto_cause_delay_trains(dataset):
    # Carica e popola l'ontologia
    ontology = load_train_data_and_populate_ontology(dataset)
    
    # Analizza i ritardi e genera il rapporto
    delay_causes = analyze_delays(ontology)
    generate_report(delay_causes)