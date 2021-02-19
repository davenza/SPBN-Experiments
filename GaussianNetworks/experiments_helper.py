DATASETS = ["ecoli70", "magic_niab", "magic_irri", "arth150"]
INSTANCES = [200, 2000, 10000]
TRAINING_FOLDS = [10]
PATIENCE = [0, 5]
SEED=0

def shd(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    shd_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        if est_arc not in true.arcs():
            shd_value += 1
            s, d = est_arc
            if (d, s) in true_arc:
                true_arcs.remove((d, s))

    for true_arc in true_arcs:
        if true_arc not in estimated_arcs:
            shd_value += 1

    return shd_value

def hamming(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    hamming_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        if est_arc not in true.arcs():
            s, d = est_arc
            if (d, s) in true_arcs:
                true_arcs.remove((d,s))
            else:
                hamming_value += 1

    for true_arc in true_arcs:
        if true_arc not in estimated_arcs:
            hamming_value += 1

    return hamming_value

def hamming_type(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    hamming_value = 0

    for n in true.nodes():
        if estimated.node_type(n) == NodeType.CKDE:
            hamming_value += 1

    return hamming_value

def remove_bidirected(pdag):
    arcs = pdag.arcs()
    bidirected_arcs = []
    
    for arc in arcs:
        if arc[::-1] in arcs:
            bidirected_arcs.append(arc)

            arcs.remove(arc)
            arcs.remove(arc[::-1])

    for to_remove in bidirected_arcs:
        pdag.remove_arc(to_remove[0], to_remove[1])

    return pdag.to_dag()