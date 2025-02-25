% Caso ricorsivo: confronta il ritardo del primo treno con il massimo ritardo calcolato per il resto della lista.
treno_piu_in_ritardo([Features|AltriTreni], Treno, Ritardo) :-
    predire_ritardo(Features, RitardoCorrente),
    treno_piu_in_ritardo(AltriTreni, TempTrain, TempDelay),
    (RitardoCorrente > TempDelay ->
        Treno = Features, Ritardo = RitardoCorrente
    ;
        Treno = TempTrain, Ritardo = TempDelay
    ).

% Caso base: se c è un solo treno, restituisci le sue caratteristiche e il ritardo predetto.
treno_piu_in_ritardo([Features], Features, Ritardo) :-
    predire_ritardo(Features, Ritardo).

% treno_piu_in_ritardo([], [], 0). Mi sa che non serve.

ritardo_conveniente(Features, RitardoProposto) :-
    predire_ritardo(Features, RitardoPredetto),
    RitardoProposto < RitardoPredetto.

% Predizione del ritardo (modificato per usare un albero decisionale simile a quello del tuo amico)
predire_ritardo(Features, Ritardo) :-
    percorri_albero(0, Features, Ritardo).

% Passo base
percorri_albero(NodeID, _, Predizione) :-
    leaf(NodeID, Predizione).

% Passo ricorsivo
percorri_albero(NodeID, Features, Predizione) :-
    node(NodeID, FeatureIndex, Threshold, LeftChild, RightChild),
    nth0(FeatureIndex, Features, FeatureValue),
    (FeatureValue =< Threshold ->
        percorri_albero(LeftChild, Features, Predizione)
    ;
        percorri_albero(RightChild, Features, Predizione)
    ).
