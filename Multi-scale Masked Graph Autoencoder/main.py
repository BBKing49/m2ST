import logging
import yaml
import numpy as np
from tqdm import tqdm
import torch
import pickle
import scipy
from scipy import sparse
from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
)
from graphmae.models import build_model
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.metrics.cluster import normalized_mutual_info_score, \
    adjusted_rand_score
from sklearn.metrics import confusion_matrix, davies_bouldin_score, silhouette_score
from torch_geometric.data import Data, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    if merge:
        cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = silhouette_score(X_embedding, cluster_labels, metric='euclidean')

    return cluster_labels, score

def get_data(dataset):
    data_folder = 'generated_data/'+ dataset +'/'
    with open(data_folder + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_folder + 'features.npy')
    lambda_I = 0.3
    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1-lambda_I)*adj_0 + lambda_I*adj_I
    if dataset not in ['mouse_hippocampus', 'mouse_olfactory_bulb','HN','HP']:
        cell_batch_info = np.load(data_folder + 'labels.npy', allow_pickle=True)
        return adj_0, adj, X_data, cell_batch_info
    # edge_dta = np.load(data_folder + 'all_distance_matrix.npy')
    else:
        return adj_0, adj, X_data

def get_graph(adj, X):
    # create sparse matrix
    row_col = []
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz()
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
    graph_bags.append(graph)
    return graph_bags



def pretrain(model, data_loader, optimizer, max_epoch, device, scheduler, logger=None):
    logging.info("start training..")

    epoch_iter = tqdm(range(max_epoch))


    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for subgraph in data_loader:
            subgraph = subgraph.to(device)
            loss, loss_dict = model(subgraph, subgraph.ndata["feat"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    return model


def main(args, dataset):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds

    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer

    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    weight_decay_f = args.weight_decay_f
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging


    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        scheduler = None
        if dataset not in ['mouse_hippocampus', 'mouse_olfactory_bulb','HN','HP']:
            adj_0, adj, X_data, cell_type_indeces = get_data(dataset)
            cell_type_indeces = cell_type_indeces.astype(int)
            n_clusters = max(cell_type_indeces) #+ 1
        else:
            adj_0, adj, X_data = get_data(dataset)
            n_clusters = 10
        num_cell = X_data.shape[0]
        num_feature = X_data.shape[1]
        args.num_features = num_feature


        df = dgl.from_scipy(scipy.sparse.coo_matrix(adj))
        df.ndata['feat'] = torch.tensor(X_data).to(torch.float)
        df_list = [df]
        data_loader = GraphDataLoader(df_list, batch_size=1)


        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if not load_model:
            model = pretrain(model, data_loader, optimizer, max_epoch, device, scheduler, logger)


        # if load_model:
        #     logging.info("Loading Model ... ")
        #     model.load_state_dict(torch.load("checkpoint.pt"))
        # if save_model:
        #     logging.info("Saveing Model ...")
        #     torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        for subgraph in data_loader:
            subgraph.to(device)
            subgraph = subgraph.to(device)
            feat = subgraph.ndata["feat"]

            X_embedding, X_embedding1 = model.embed(subgraph, feat) #改成多组

            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding1 = X_embedding1.cpu().detach().numpy()

            scipy.io.savemat('embedding/'+dataset+'_embded.mat',
                             {'view1': X_embedding, 'view2': X_embedding1, 'labels': cell_type_indeces})



def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    import time
    start_time = time.time()
    args = build_args()
    # if args.use_cfg:
    #     args = load_best_configs(args, "configs.yml")
    print(args)
    dataset = 'STARmap'
    main(args, dataset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.4f} 秒")
