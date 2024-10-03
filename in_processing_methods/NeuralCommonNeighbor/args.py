import argparse
from model import predictor_dict, convdict


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--runs', type=int, default=3, help="number of repeated runs")
    parser.add_argument('--device', type=int, default=0, help="which device to use to run the process")
    parser.add_argument('--dataset', type=str, default="collab")
    
    parser.add_argument('--batch_size', type=int, default=8192, help="batch size")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hiddim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--gnnlr', type=float, default=0.0003, help="learning rate of gnn")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    
    parser.add_argument('--adv_lr', type=float, default=0.0001, help="learning rate of fairness model")
    parser.add_argument('--reg_lambda', type=float, default=0.0001, help="learning rate of fairness model")
    
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")

    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    # NCN can sample common neighbors for scalability. Generally not used. 
    parser.add_argument('--cndeg', type=int, default=-1)
    
    # predictor used, such as NCN, NCNC
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.
    parser.add_argument('--model', choices=convdict.keys())

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")
    
    parser.add_argument("--no_intervention", action="store_true", help="removes the intervention model")
    
    parser.add_argument("--link_level", action="store_true", help="link level fair model prediction")
    
    parser.add_argument("--node_split", action="store_true", help="Instead of enforcing link fairness, it enforces node fairness")
    
    parser.add_argument("--laftr_dp", action="store_true", help="Implements LAFTR loss for Demografic Parity")
    parser.add_argument("--laftr_eo", action="store_true", help="Implements LAFTR loss for Equality of Opportunity")
    
    parser.add_argument("--dp_only", action="store_true", help="Removes the link prediction loss from the training process")
    
    parser.add_argument("--nifty", action="store_true", help="Implements NIFTY counterfactual approach in NCN")
    
    parser.add_argument("--no_wandb", action="store_true", help="no wandb")
    parser.add_argument("--wandb_sweep", action="store_true", help="wandb sweep")
    
    parser.add_argument("--fairsample", action="store_true", help="whether to sample edges fairly")
    
    parser.add_argument("--node_classification", action="store_true", help="uses the fair embeddings for node classification task.")
    parser.add_argument("--node_classifier_num_layers", type=int, default=3, help="Num. layers for node classification.")
    
    parser.add_argument("--inference", action="store_true", help="Only inference on nodes")
    
    parser.add_argument("--fair_learner_gnn", action="store_true", help="Use GNN as fairlearner")
    parser.add_argument("--alternate_training", action="store_true", help="Alternate training between GNN and Fairness model")
    
    parser.add_argument("--gp_coef", type=float, default=0.01)
    parser.add_argument("--concat", action="store_true", help="concatenate the node embeddings")
    parser.add_argument("--random_features", action="store_true", help="concatenate the node embeddings")
    
   
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    args = parser.parse_args()
    return args