% Passo ricorsivo: Prende in input una lista di features e salva man mano il ritardo più alto
treno_piu_in_ritardo([Features|AltriTreni], Treno, Ritardo) :-
    predire_ritardo(Features, RitardoCorrente),
    treno_piu_in_ritardo(AltriTreni, MaxTreno, MaxRitardo),
    (RitardoCorrente > MaxRitardo ->
     Treno = Features, Ritardo = RitardoCorrente
     ;
     Treno = MaxTreno, Ritardo = MaxRitardo).

% Passo base: se c è un solo treno, restituisci le sue caratteristiche e il ritardo predetto.
treno_piu_in_ritardo([Features], Features, Ritardo) :-
    predire_ritardo(Features, Ritardo).

% treno_piu_in_ritardo([], [], 0). Dava un errore quindi ho cambiato il passo base con quello che sta sopra

% Predizione del ritardo più conveniente
ritardo_conveniente(Features, RitardoProposto) :-
    predire_ritardo(Features, RitardoPredetto),
    RitardoProposto < RitardoPredetto.

% Predizione del ritardo
predire_ritardo(Features, Ritardo) :-
    reverse(Features, FeaturesInvertite),
    calcolare_incremento(FeaturesInvertite, 0, Ritardo).

% Passo ricorsivo: Predicato per calcolare gli incrementi basati sulle caratteristiche
calcolare_incremento([Feature|AltreFeature], RitardoParziale, Ritardo) :-
    length(AltreFeature, Indice),
    incremento_feature(Indice, Feature, Incremento),
    NuovoRitardoParziale is RitardoParziale + Incremento,
    calcolare_incremento(AltreFeature, NuovoRitardoParziale, Ritardo).

% Passo base: Se non ci sono più feature da analizzare, restituisci il ritardo parziale
calcolare_incremento([], RitardoParziale, RitardoParziale).

% Di seguito gli incrementi basati sulle caratteristiche

% Year (Anni più recenti -> meno ritardo)
incremento_feature(0, Anno, Incremento) :-
    Incremento is (2025 - Anno) * -0.5.

% Month (Inverno -> Più ritardo)
incremento_feature(1, Mese, Incremento) :-
    (Mese >= 11 ; Mese =< 2 -> Incremento is 5 ; Incremento is 0).

% Average travel time (Più viaggio = Più probabilità di ritardo)
incremento_feature(2, TempoMedio, Incremento) :-
    Incremento is TempoMedio * 0.1.

% Number of cancelled trains (Più cancellazioni = Maggiore congestione e ritardo)
incremento_feature(4, Cancellati, Incremento) :-
    Incremento is Cancellati * 1.

% Number of late trains at departure (Più treni in ritardo alla partenza = Più ritardi in arrivo)
incremento_feature(5, TardiPartenza, Incremento) :-
    Incremento is TardiPartenza * 0.8.

% Average delay of late departing trains (Aumenta direttamente il ritardo previsto)
incremento_feature(6, RitardoPartenza, Incremento) :-
    Incremento is RitardoPartenza * 0.9.

% % trains late due to external causes (Più cause esterne = Più ritardo)
incremento_feature(11, PercentualeCauseEsterne, Incremento) :-
    Incremento is PercentualeCauseEsterne * 0.5.

% Number of late trains > 15min (Influenza direttamente il ritardo)
incremento_feature(17, Tardi15Min, Incremento) :-
    Incremento is Tardi15Min * 1.5.

% Number of late trains > 30min (Influenza più pesantemente il ritardo)
incremento_feature(19, Tardi30Min, Incremento) :-
    Incremento is Tardi30Min * 2.

% Default per le feature non specificate
incremento_feature(_, _, 0).
