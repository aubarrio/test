from src.model import *
from src.utils import *

def complete(fp, source, hidden, dropout, lr, weight_decay, lam, epochs):
    if source == "cora":
        data, edges = read_cora_data(fp)
    elif source == "twitch":
        data, edges = read_twitch_data(fp)
    elif source == "facebook":
        data, edges = read_facebook_data(fp)

    labels, idx, X = parse_data(data)

    features = build_features(X)

    labels = encode_label(labels)

    edges = build_edges(idx, parse_edges(edges))

    adj = build_adj(edges, labels)

    labels_for_lpa = torch.from_numpy(labels).type(torch.FloatTensor)

    labels = torch.LongTensor(np.where(labels)[1])

    # idx_train, idx_val, idx_test = build_idx()
    idx_train, idx_val, idx_test = build_idx(X.shape[0])

    model = GCNLPA(nfeat=features.shape[1],
              nhid=hidden,
              nclass=labels.max().item() + 1,
              adj=adj,
              dropout_rate=dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        output, y_hat = model(features, adj, labels_for_lpa)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        loss_lpa = F.nll_loss(y_hat, labels)
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train = loss_gcn + lam * loss_lpa
        loss_train.backward(retain_graph=True)
        optimizer.step()
        model.eval()
        output_val, _ = model(features, adj, labels_for_lpa)
        loss_val = F.nll_loss(output_val[idx_val], labels[idx_val])
        acc_val = accuracy(output_val[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(i+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

    model.eval()
    output, _ = model(features, adj, labels_for_lpa)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
