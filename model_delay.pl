% Prende in input una lista di features e salva man mano il ritardo più alto
treno_piu_in_ritardo([Features|AltriTreni], Treno, Ritardo) :-
    predire_ritardo(Features, RitardoCorrente),
    treno_piu_in_ritardo(AltriTreni, MaxTreno, MaxRitardo),
    (RitardoCorrente > MaxRitardo ->
     Treno = Features, Ritardo = RitardoCorrente
     ;
     Treno = MaxTreno, Ritardo = MaxRitardo).

treno_piu_in_ritardo([], [], 0).

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
