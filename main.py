import argparse
import os
import torch

import matplotlib.pyplot as plt

from util import load_node_csv, load_edge_csv
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.models import LightGCN

from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
from torch import optim

from metric import get_metrics


def run(args):
    user_mapping = load_node_csv(args.data_path, index_col='MAC')
    item_mapping = load_node_csv(args.data_path, index_col='ptitle', offset=len(user_mapping))
    edge_index = load_edge_csv(
        args.data_path,
        src_index_col='MAC',
        src_mapping=user_mapping,
        dst_index_col='ptitle',
        dst_mapping=item_mapping,
    )
    
    num_users, num_items = len(user_mapping), len(item_mapping)
    num_edges = edge_index.shape[1]
    all_indices = [i for i in range(num_edges)]

    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)
    train_edge_index = edge_index[:, train_indices]
    test_edge_index = edge_index[:, test_indices]

    train_loader = DataLoader(train_edge_index.T, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_edge_index.T, batch_size=1024, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightGCN(
        num_nodes=num_users+num_items,
        embedding_dim=64,
        num_layers=3,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.num_epochs
    topk = args.topk

    train_loss_list = []
    test_loss_list = []
    recall_list = []
    precision_list = []
    ndcg_list = []

    for epoch in range(num_epochs):
        train_loss = 0.

        model.train()
        for batch in train_loader:
            pos_edges = batch.T.to(device)
            neg_edges = negative_sampling(train_edge_index, num_nodes=[num_users, num_items], num_neg_samples=pos_edges.shape[1]).to(device)

            pos_rank = model.forward(pos_edges)
            neg_rank = model.forward(neg_edges)

            loss = model.recommendation_loss(pos_rank, neg_rank)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss/len(train_loader)

        test_loss = 0.
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                pos_edges = batch.T.to(device)
                neg_edges = negative_sampling(test_edge_index, num_nodes=[num_users, num_items], num_neg_samples=pos_edges.shape[1]).to(device)

                pos_rank = model.forward(pos_edges)
                neg_rank = model.forward(neg_edges)

                loss = model.recommendation_loss(pos_rank, neg_rank)

                test_loss += loss.item()
            
            test_loss = test_loss/len(test_loader)
            recall, precision, ndcg = get_metrics(model, test_edge_index.to(device), train_edge_index.to(device), num_users, topk)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        recall_list.append(recall)
        precision_list.append(precision)
        ndcg_list.append(ndcg)

        print(f"[Epoch {epoch}/{num_epochs}] \ 
        train_loss: {round(train_loss, 5)}, \
        test_loss: {round(test_loss, 5)}, \
        test_recall@{topk}: {round(recall, 5)}, \
        test_precision@{topk}: {round(precision, 5)}, \
        test_ndcg@{topk}: {round(ndcg, 5)}")
    
    plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
    plt.plot(range(len(test_loss_list)), test_loss_list, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curves')
    plt.savefig('loss')
    plt.clf()

    plt.plot(range(len(recall_list)), recall_list, label='recall')
    plt.plot(range(len(precision_list)), precision_list, label='precision')
    plt.plot(range(len(ndcg_list)), ndcg_list, label='ndcg')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('test score curves')
    plt.savefig('score')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparse LightGCN')
    parser.add_argument('--data_path', default='./data/kobaco.csv')
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--topk', default=10)
    args = parser.parse_args()

    run(args)