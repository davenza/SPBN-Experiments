import glob
from pathlib import Path
from pybnesian import load
from pybnesian.factors import NodeType
from pybnesian.models import BayesianNetworkType, BayesianNetworkBase
import networkx as nx
import subprocess

def draw_model(model, filename):
    
    DG = nx.DiGraph()
    DG.add_nodes_from(model.nodes())
    DG.add_edges_from(model.arcs())

    if isinstance(m, BayesianNetworkBase) and model.type == BayesianNetworkType.Semiparametric:
        for node in DG.nodes:
            if model.node_type(node) == NodeType.CKDE:
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'

    a = nx.nx_agraph.to_agraph(DG)
    if filename[-4:] != '.dot':
        filename += '.dot'
    a.write(filename)
    a.clear()

    pdf_out = filename[:-4] + '.pdf'
    subprocess.run(["dot", "-Tpdf", filename, "-o", pdf_out])


all_models = glob.glob('models/**/*.pickle', recursive=True)
for model in all_models:
    print(model)
    m = load(model)
    filename = model[:-7] + '.dot'
    draw_model(m, filename)