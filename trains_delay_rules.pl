% Passo ricorsivo: Prende in input una lista di features e salva man
% mano il ritardo piu' alto
treno_piu_in_ritardo([Features|AltriTreni], Treno, Ritardo) :-
    predire_ritardo(Features, RitardoCorrente),
    treno_piu_in_ritardo(AltriTreni, MaxTreno, MaxRitardo),
    (RitardoCorrente > MaxRitardo ->
     Treno = Features, Ritardo = RitardoCorrente
     ;
     Treno = MaxTreno, Ritardo = MaxRitardo).

% Passo base: se c'e' un solo treno, restituisci le sue caratteristiche
% e il ritardo predetto.
treno_piu_in_ritardo([Features], Features, Ritardo) :-
    predire_ritardo(Features, Ritardo).

% Predizione del ritardo piu' conveniente
ritardo_conveniente(Features, RitardoProposto, RitardoPredetto, Conveniente) :-
    predire_ritardo(Features, RitardoPredetto),
    (RitardoProposto < RitardoPredetto ->
        Conveniente = true
    ;
        Conveniente = false).

% Predizione del ritardo
predire_ritardo(Features, Ritardo) :-
    reverse(Features, FeaturesInvertite),
    calcolare_ritardo(FeaturesInvertite, 0, Ritardo).

% Passo ricorsivo: Predicato per calcolare gli incrementi basati sulle caratteristiche
calcolare_ritardo([Feature|AltreFeature], RitardoParziale, Ritardo) :-
    length(AltreFeature, Indice),
    incremento_feature(Indice, Feature, Incremento),
    NuovoRitardoParziale is RitardoParziale + Incremento,
    calcolare_ritardo(AltreFeature, NuovoRitardoParziale, Ritardo).

% Passo base: Se non ci sono piu' feature da analizzare, restituisci il
% ritardo parziale
calcolare_ritardo([], RitardoParziale, RitardoParziale).

% Di seguito gli incrementi basati sulle caratteristiche specifiche
% Year
% Month
% Number of expected circulations
% Number of cancelled trains
% Number of late trains at departure
% Average travel time
% Average delay of late departing trains

% Year (Anni piu' recenti -> meno ritardo)
incremento_feature(0, Anno, Incremento) :-
    Incremento is (2025 - Anno) * -0.5.

% Month (Inverno -> Piu' ritardo)
incremento_feature(1, Mese, Incremento) :-
    (Mese >= 11 ; Mese =< 2 -> Incremento is 5 ; Incremento is 0).

% Indice 4: Number of expected circulations
incremento_feature(2, Circolazioni, Incremento) :-
    Incremento is Circolazioni * 0.2.

% Number of cancelled trains (Piu' cancellazioni = Maggiore congestione
% e ritardo)
incremento_feature(3, Cancellati, Incremento) :-
    Incremento is Cancellati * 1.

% Number of late trains at departure (Piu' treni in ritardo alla
% partenza = Piu' ritardi in arrivo)
incremento_feature(4, NRitardiPartenza, Incremento) :-
    Incremento is NRitardiPartenza * 0.8.

% Average travel time (Piu' viaggio = Piu' probabilita' di ritardo)
incremento_feature(5, TempoMedio, Incremento) :-
    Incremento is TempoMedio * 0.1.

% Average delay of late departing trains (Aumenta direttamente il ritardo previsto)
incremento_feature(6, RitardoPartenza, Incremento) :-
    Incremento is RitardoPartenza * 0.9.

% Default per le feature non specificate
incremento_feature(_, _, 0).
