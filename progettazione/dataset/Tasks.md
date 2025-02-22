# Task di Progettazione

## Predizione

- **Obiettivo**: Predire il ritardo medio (in minuti) di tutti i treni in arrivo.

- **Caratteristiche per la Predizione**:

  - **Average travel time (min)**: Tempo medio di viaggio dei treni.
  - **Number of cancelled trains**: Numero di treni cancellati.
  - **Number of late trains at departure**: Numero di treni in partenza in ritardo.
  - **Number of trains late on arrival**: Numero di treni che arrivano in ritardo.
  - **Average delay of late departing trains (min)**: Ritardo medio dei treni in partenza in ritardo.
  - **Average delay of all departing trains (min)**: Ritardo medio di tutti i treni in partenza.
  - **Average delay of late arriving trains (min)**: Ritardo medio dei treni che arrivano in ritardo.
  - **% trains late due to external causes**: Percentuale di treni in ritardo a causa di cause esterne (maltempo, ostacoli, ecc.).
  - **% trains late due to railway infrastructure**: Percentuale di treni in ritardo a causa dell'infrastruttura ferroviaria.
  - **% trains late due to traffic management**: Percentuale di treni in ritardo a causa della gestione del traffico ferroviario.
  - **% trains late due to rolling stock**: Percentuale di treni in ritardo a causa di problemi con i veicoli.
  - **% trains late due to station management and reuse of material**: Percentuale di treni in ritardo a causa della gestione della stazione.
  - **% trains late due to passenger traffic**: Percentuale di treni in ritardo a causa del traffico passeggeri.
  - **Number of late trains > 15min**: Numero di treni con ritardo superiore ai 15 minuti.
  - **Average train delay > 15min**: Ritardo medio dei treni con ritardo superiore ai 15 minuti.
  - **Number of late trains > 30min**: Numero di treni con ritardo superiore ai 30 minuti.
  - **Number of late trains > 60min**: Numero di treni con ritardo superiore ai 60 minuti.
  - **Year**: Anno di riferimento.
  - **Month**: Mese di riferimento.

- **Target**:
  - **Average delay of all arriving trains (min)**: Ritardo medio di tutti i treni in arrivo.

## Prolog e KB

- **Obiettivo**: Predire il **ritardo medio dei treni in arrivo** utilizzando un modello basato su regole e incrementi. Il sistema prende in input le caratteristiche di un treno e stima il **"Average delay of all arriving trains (min)"**, permettendo di individuare il treno con il ritardo maggiore e di valutare se un ritardo proposto è conveniente.

- **Caratteristiche per la Predizione**:

  1. **Year**: Anno di riferimento.
  2. **Month**: Mese (per cogliere la stagionalità).
  3. **Number of expected circulations**: Numero di circolazioni previste (treni programmati).
  4. **Number of cancelled trains**: Numero di treni cancellati.
  5. **Number of late trains at departure**: Numero di treni in ritardo alla partenza.
  6. **Average delay of late departing trains (min)**: Ritardo medio dei treni in ritardo alla partenza.
  7. **% trains late due to external causes**: Percentuale di treni in ritardo per cause esterne.
  8. **% trains late due to railway infrastructure**: Percentuale di ritardi causati da problemi infrastrutturali.
  9. **% trains late due to traffic management**: Percentuale di ritardi per la gestione del traffico.
  10. **% trains late due to rolling stock**: Percentuale di ritardi dovuti a problemi con il materiale rotabile.
  11. **% trains late due to passenger traffic**: Percentuale di ritardi causati dal traffico passeggeri.

- **Target**:
  **Average delay of all arriving trains (min)**: È la variabile di uscita (target) che il modello predice, ovvero il **ritardo medio di tutti i treni in arrivo**.

## Apprendimento Bayesiano

- **Obiettivo**: Date le caratteristiche di un treno (eccetto la target), stimare la probabilità di una delle caratteristiche.

- **Caratteristiche per la Predizione**:

  1. **Month** (Mese dell'anno)
  2. **Average travel time (min)** (Tempo medio di viaggio)
  3. **% trains late due to external causes**: Percentuale di treni in ritardo per cause esterne.
  4. **% trains late due to railway infrastructure**: Percentuale di ritardi causati da problemi infrastrutturali.
  5. **% trains late due to traffic management**: Percentuale di ritardi per la gestione del traffico.
  6. **% trains late due to rolling stock**: Percentuale di ritardi dovuti a problemi con il materiale rotabile.
  7. **% trains late due to station management and reuse of material**: Percentuale di ritardi dovuti alla gestione della stazione e al riutilizzo del materiale.
  8. **% trains late due to passenger traffic**: Percentuale di ritardi causati dal traffico passeggeri.

- **Target**: Una delle caratteristiche.

---
