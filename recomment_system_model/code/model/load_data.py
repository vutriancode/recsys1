from CONFIG import *

class Training:

    def __init__(self, model):
        self.model = model

    def train(self, train_loader, optimizer, epoch, best_rmse, best_mae, device="cpu"):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            batch_nodes_u, batch_nodes_v, labels_list = data
            optimizer.zero_grad()
            loss = self.model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            if i % 1 == 0:
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch, i, running_loss / 1, best_rmse, best_mae))
                running_loss = 0.0
        return 0