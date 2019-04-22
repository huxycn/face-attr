import os
import re
import time
import torch as t
from torch import nn

from torch.utils.data import DataLoader

from utils import progress_bar
from utils import Visualizer

device = 'cuda' if t.cuda.is_available() else 'cpu'

vis = Visualizer(env='face_attr')

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        class_name = str(type(self))
        pattern = re.compile('\'(.*)\'')
        self.model_name = pattern.findall(class_name)[0]

    def compile(self, lr, epochs, criterion, optimizer):
        self.lr = lr
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer

    def __adjust_learning_rate(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(self, train_loader, val_loader):
        # 训练模式
        self.train()
        for epoch in range(self.epochs):
            print('Epoch: {}/{}'.format(epoch+1, self.epochs))
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                msg = '- loss: {:.4f} - acc: {:.4f}'.format(train_loss / total, correct / total)
                progress_bar(batch_idx, len(train_loader), msg, end='\r')

            vis.plot('train_loss', train_loss / total)
            vis.plot('train_acc', correct / total)

            val_loss, val_acc = self.evaluate(val_loader, prefix='val')
            vis.plot('val_loss', val_loss)
            vis.plot('val_acc', val_acc)
            msg = '- loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}'.format(train_loss / total,
                                                                                             correct / total,
                                                                                             val_loss, val_acc)
            progress_bar(len(train_loader) - 1, len(train_loader), msg, end='\n')
            if epoch % 10 == 9:
                self.save(epoch, self.epochs)
            self.__adjust_learning_rate(epoch)

        pass

    def evaluate(self, eval_loader, prefix='eval'):
        # 测试模式
        self.eval()
        loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self(inputs)
            loss += self.criterion(outputs, targets).item()
            total += targets.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        loss = loss / total
        acc = correct / total
        return loss, acc

    def predict(self, x):
        pass

    def save(self, epoch, epochs):
        checkpoints_dir = 'checkpoints'
        checkpoint_name = '{model_name}_epoch[{epoch}.{epochs}]_{timestamp}.pt'.format(
            model_name=self.model_name,
            epoch=epoch,
            epochs=epochs,
            timestamp=time.strftime('%y%m%d%H%M%S')
        )
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        checkpoint_fp = os.path.join(checkpoints_dir, checkpoint_name)
        if not os.path.exists(checkpoint_fp):
            os.mknod(checkpoint_fp)
        t.save(self.state_dict(), checkpoint_fp)

    def load(self, path):
        self.load_state_dict(t.load(path))
