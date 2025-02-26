% Passo ricorsivo: Trova il treno con il ritardo massimo in una lista.
treno_piu_in_ritardo([Features|AltriTreni], Treno, Ritardo) :-
    predire_ritardo(Features, RitardoCorrente),
    treno_piu_in_ritardo(AltriTreni, TempTrain, TempDelay),
    (RitardoCorrente > TempDelay ->
        Treno = Features, Ritardo = RitardoCorrente
    ;
        Treno = TempTrain, Ritardo = TempDelay
    ).

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
    percorri_albero(0, Features, Ritardo).

% Passo base (nodo foglia)
percorri_albero(NodeID, _, Predizione) :-
    leaf(NodeID, Predizione).

% Passo ricorsivo (nodo intermedio)
percorri_albero(NodeID, Features, Predizione) :-
    node(NodeID, FeatureIndex, Threshold, LeftChild, RightChild),
    nth0(FeatureIndex, Features, FeatureValue),
    (FeatureValue =< Threshold ->
        percorri_albero(LeftChild, Features, Predizione)
    ;
        percorri_albero(RightChild, Features, Predizione)
    ).
