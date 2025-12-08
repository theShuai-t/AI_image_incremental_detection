import copy
import logging
import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from convs.vit_adapter_c import vit_adapter_patch16_224
# from convs.vit_adapter_raw import vit_adapter_patch16_224
# from convs.vit_adapter_dynamic import vit_adapter_patch16_224
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, load_model, early_stop, set_random, metrics_others
from utils.triple_loss import TripletLoss, RelaxTripletLoss


EPSILON = 1e-8
init_epochs = 1
update_epochs = 1
init_lrate = 2e-3
lrate = 2e-3
init_step_size = 10
update_step_size = 10
lrate_decay = 0.5
batch_size = 32
weight_decay = 2e-4
num_workers = 8
T = 2
cos_loss_weight = 0.1
center_loss_weight = 0.001
tri_loss_weight = 0.1
multi = 1
rkd_T = 1

start_center = 20
rkd_loss_weight = 5
kd_loss_weight = 1

print('start_center:', start_center, 'rkd_loss_weight:', rkd_loss_weight, "rkd_T:", rkd_T)


def build_net():
    model = vit_adapter_patch16_224(False, num_classes=2)
    model.load_pretrained(
        checkpoint_path='/home/tangshuai/.cache/huggingface/hub/vit_base_patch16_224/imagenet2012.npz')
    for name, p in model.named_parameters():
        if 'adapter' in name or 'head' in name or 'shuffle' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def get_old_net(model):
    old_model = copy.deepcopy(model)
    for p in old_model.parameters():
        p.requires_grad = False
    old_model.eval()
    return old_model


class Vit_Xception_Center(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_network = None
        self._network = build_net()
        self.best_model_path = []

    def after_task(self):
        self._old_network = get_old_net(self._network)
        self._known_classes = self._total_classes

        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        if self._cur_task == 0:
            shot = None
        else:
            shot = self.args["shot"]
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            resize_size=self.args["resize_size"],
            appendent=self._get_memory(),
            shot=shot
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="val", mode="val", resize_size=self.args["resize_size"]
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test", resize_size=self.args["resize_size"]
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        out_dataset = data_manager.get_dataset(
            None, source="test_out", mode="test", resize_size=self.args["resize_size"]
        )
        self.out_loader = DataLoader(
            out_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        if self._cur_task == 0:
            if self.args['skip']:
                state = load_model(self._network, self.args, self.best_model_path)
                self._network.to(self._device)
            else:
                self._train(self.train_loader, self.val_loader)
        else:
            self._train(self.train_loader, self.val_loader)
        self._eval(self.test_loader, self.out_loader, data_manager)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            e_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=init_lrate,
                                     weight_decay=weight_decay)
            e_scheduler = optim.lr_scheduler.StepLR(optimizer=e_optimizer, step_size=init_step_size, gamma=lrate_decay)
        else:
            e_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), lr=lrate,
                                     weight_decay=weight_decay)
            e_scheduler = optim.lr_scheduler.StepLR(optimizer=e_optimizer, step_size=update_step_size,
                                                    gamma=lrate_decay)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            self._init_train(train_loader, val_loader, e_optimizer, e_scheduler)
        else:
            self._update_representation(train_loader, val_loader, e_optimizer, e_scheduler)

    def progressive_update_center(self, center, current_center):
        if center is None:
            center = current_center
        else:
            sim = ((F.cosine_similarity(center, current_center) + 1) / 2).unsqueeze(1)
            center = (sim**2) * current_center + (1 - sim**2) * center
        return center

    def category_center_init(self, model, old=False):
        real_features = None
        model.eval()
        for i, (_, inputs, targets, order) in enumerate(self.train_loader):
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                _, out = model(inputs[targets])
                features = out['features'][:, 0].view(inputs.size()[0], -1)
                if real_features is not None:
                    _features = features[targets == 0]
                    real_features = torch.cat([real_features, _features], dim=0)
                else:
                    real_features = features[targets == 0]
        real_features = torch.mean(real_features, dim=0).unsqueeze(0)
        if not old:
            model.train()
        return real_features

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        rtriple_loss = RelaxTripletLoss()
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(init_epochs))
        criterion_cosine = nn.CosineEmbeddingLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            cos_losses = 0.0
            center_losses = 0.0
            tri_losses = 0.0
            correct, total = 0, 0

            apply_c = True if (epoch + 1) >= start_center else False
            if apply_c:
                epoch_real_global_center = self.category_center_init(self._network)
                each_batch_real_global_center = None

            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                order = order.to(self._device)
                logits, out = self._network(inputs)
                features = out['features'][:, 0].view(inputs.size()[0], -1)
                real_f = features[targets == 0]
                fake_f = features[targets == 1]
                bs = inputs.shape[0]

                cls_loss = F.cross_entropy(logits, targets)

                shuffle_net = out['shuffle_net']
                cos_loss_1 = criterion_cosine(shuffle_net['x_no_1'].view(bs, -1),
                                              shuffle_net['x_shuffle_1'].view(bs, -1), torch.ones(bs).to(self._device))
                cos_loss_2 = criterion_cosine(shuffle_net['x_no_2'].view(bs, -1),
                                              shuffle_net['x_shuffle_2'].view(bs, -1), torch.ones(bs).to(self._device))
                cos_loss = cos_loss_1 + cos_loss_2

                targets_c = copy.deepcopy(targets)
                for j in range(len(targets)):
                    if targets[j] == 1:
                        targets_c[j] = order[j] + 1

                tri_loss = rtriple_loss(features, targets_c, self._device)
                loss = cls_loss + cos_loss * cos_loss_weight + tri_loss * tri_loss_weight

                if apply_c:
                    current_total_center = torch.mean(real_f, dim=0).unsqueeze(0)
                    if i == 0:
                        each_batch_real_global_center = current_total_center.clone().detach()
                    else:
                        each_batch_real_global_center = torch.cat([each_batch_real_global_center, current_total_center.clone().detach()], dim=0)
                    current_total_center_ = torch.mean(each_batch_real_global_center, dim=0).unsqueeze(0)
                    batch_real_global_center = self.progressive_update_center(epoch_real_global_center, current_total_center_)
                    epoch_real_global_center = batch_real_global_center.clone().detach()
                    f = self.center_l(fake_f, batch_real_global_center)
                    r = self.center_l(real_f, batch_real_global_center)
                    f2r = (torch.max(torch.zeros_like(f), torch.max(r) - f)).sum() / bs
                    center_loss = r.sum() / bs + multi * f2r
                    loss = loss + center_loss * center_loss_weight
                else:
                    center_loss = torch.tensor(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cls_losses += cls_loss.item()
                cos_losses += cos_loss.item()
                center_losses += center_loss.item()
                tri_losses += tri_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            # center_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = self._compute_accuracy(self._network, val_loader)
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
                                            self._cur_task, self._network)
            info = "Task {}, Epoch {}/{} => cls {:.4f} ,cos {:.4f}, center {:.4f}, tri {:.4f}, train_acc {:.2f}, val_acc {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                init_epochs,
                cls_losses / len(train_loader),
                cos_losses / len(train_loader),
                center_losses / len(train_loader),
                tri_losses / len(train_loader),
                # center_dist_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
                # center_optimizer.param_groups[0]['lr'],
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _rkd(self, features, old_features, centers, old_centers, weights=None):
        def get_sim_map(feature, center):
            # return torch.cosine_similarity(features, center, dim=1)
            bs = feature.size(0)
            distmat = torch.pow(feature, 2).sum(dim=1, keepdim=True).expand(bs, 1) + \
                      torch.pow(center, 2).sum(dim=1, keepdim=True).expand(1, bs).t()
            distmat.addmm_(feature, center.t(), beta=1, alpha=-2)

            loss = distmat.clamp(min=1e-12, max=1e+12)
            return loss

        def get_sim_map_cos(feature, center):
            return torch.cosine_similarity(features, center, dim=1)
        s_sim_map = get_sim_map(features, centers)
        t_sim_map = get_sim_map(old_features, old_centers)
        # sim_dis = F.smooth_l1_loss(s_sim_map, t_sim_map, reduction='elementwise_mean')
        sim_dis = F.kl_div(F.log_softmax(s_sim_map / rkd_T, dim=0), F.softmax(t_sim_map / rkd_T, dim=0),
                           reduction='batchmean')

        return sim_dis

    def center_l(self, x, centers):
        batch_size = x.size(0)
        # x = F.normalize(x, dim=1)
        # centers = F.normalize(centers, dim=1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(1, batch_size).t()
        distmat.addmm_(x, centers.t(), beta=1, alpha=-2)

        loss = distmat.clamp(min=1e-12, max=1e+12)

        return loss

    def before_update(self):
        for name, p in self._network.named_parameters():
            if 'adapter' in name:
                p.requires_grad = False

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        if self._cur_task == 1:
            self.before_update()
        rtriple_loss = RelaxTripletLoss()
        triple_loss = TripletLoss()
        best_acc, patience, self.best_model_path = 0, 0, []
        prog_bar = tqdm(range(update_epochs))
        criterion_cosine = nn.CosineEmbeddingLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            cls_losses = 0.0
            cos_losses = 0.0
            correct, total = 0, 0
            kd_losses = 0.0
            rkd_losses = 0.0
            tri_losses = 0.0
            center_losses = 0.0

            apply_c = True if (epoch + 1) >= start_center else False
            if apply_c:
                epoch_real_global_center = self.category_center_init(self._network)
                old_epoch_real_global_center = self.category_center_init(self._old_network, old=True)
                each_batch_real_global_center = None
                old_each_batch_real_global_center = None

            for i, (_, inputs, targets, order) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                order = order.to(self._device)

                logits, out = self._network(inputs)

                features = out['features'][:, 0].view(inputs.size()[0], -1)
                real_f = features[targets == 0].clone()
                fake_f = features[targets == 1].clone()
                bs = inputs.shape[0]

                old_logits, old_out = self._old_network(inputs)
                old_features = old_out['features'][:, 0].view(inputs.size()[0], -1)
                old_real_f = old_features[targets == 0].clone()

                cls_loss = F.cross_entropy(logits, targets)
                kd_loss = _KD_loss(logits, old_logits, T)

                shuffle_net = out['shuffle_net']
                cos_loss_1 = criterion_cosine(shuffle_net['x_no_1'].view(bs, -1),
                                              shuffle_net['x_shuffle_1'].view(bs, -1), torch.ones(bs).to(self._device))
                cos_loss_2 = criterion_cosine(shuffle_net['x_no_2'].view(bs, -1),
                                              shuffle_net['x_shuffle_2'].view(bs, -1), torch.ones(bs).to(self._device))
                cos_loss = cos_loss_1 + cos_loss_2

                targets_c = copy.deepcopy(targets)
                for j in range(len(targets)):
                    if targets[j] == 1:
                        targets_c[j] = order[j].clone() + 1

                tri_loss = rtriple_loss(features, targets_c, self._device)

                loss = cls_loss + kd_loss * kd_loss_weight + cos_loss * cos_loss_weight + tri_loss * tri_loss_weight

                if apply_c:
                    current_total_center = torch.mean(real_f, dim=0).unsqueeze(0)
                    if i == 0:
                        each_batch_real_global_center = current_total_center.clone().detach()
                    else:
                        each_batch_real_global_center = torch.cat([each_batch_real_global_center, current_total_center.clone().detach()], dim=0)
                    current_total_center_ = torch.mean(each_batch_real_global_center, dim=0).unsqueeze(0)
                    batch_real_global_center = self.progressive_update_center(epoch_real_global_center, current_total_center_)
                    epoch_real_global_center = batch_real_global_center.clone().detach()

                    f = self.center_l(fake_f, batch_real_global_center)
                    r = self.center_l(real_f, batch_real_global_center)
                    f2r = (torch.max(torch.zeros_like(f), torch.max(r) - f)).sum() / bs
                    center_loss = r.sum() / bs + multi * f2r
                    loss = loss + center_loss * center_loss_weight

                    # center align
                    old_current_total_center = torch.mean(old_real_f, dim=0).unsqueeze(0)
                    if i == 0:
                        old_each_batch_real_global_center = old_current_total_center.clone().detach()
                    else:
                        old_each_batch_real_global_center = torch.cat([old_each_batch_real_global_center, old_current_total_center.clone().detach()], dim=0)
                    old_current_total_center_ = torch.mean(old_each_batch_real_global_center, dim=0).unsqueeze(0)
                    old_batch_real_global_center = self.progressive_update_center(old_epoch_real_global_center, old_current_total_center_)
                    old_epoch_real_global_center = old_batch_real_global_center.clone().detach()
                    rkd_loss = self._rkd(features, old_features, batch_real_global_center, old_batch_real_global_center)
                    loss = loss + rkd_loss * rkd_loss_weight

                else:
                    center_loss = torch.tensor(0)
                    rkd_loss = torch.tensor(0)

                optimizer.zero_grad()
                # center_optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cls_losses += cls_loss.item()
                cos_losses += cos_loss.item()
                kd_losses += kd_loss.item()
                center_losses += center_loss.item()
                tri_losses += tri_loss.item()
                rkd_losses += rkd_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            val_acc = self._compute_accuracy(self._network, val_loader)
            best_acc, patience = early_stop(self.args, best_acc, val_acc, patience,
                                            self.best_model_path, self._cur_task, self._network)
            # best_acc, patience = early_stop(self.args, best_acc, val_acc, patience, self.best_model_path,
            #                                 self._cur_task, self._network)
            info = "Task {}, Epoch {}/{} => cls {:.4f}, cos {:.4f}, center {:.4f}, tri {:.4f}, kd {:.4f}, rkd {:.4f}, train_acc {:.2f}, val_acc {:.2f}, lr {}".format(
                self._cur_task,
                epoch + 1,
                update_epochs,
                cls_losses / len(train_loader),
                cos_losses / len(train_loader),
                center_losses / len(train_loader),
                tri_losses / len(train_loader),
                kd_losses / len(train_loader),
                rkd_losses / len(train_loader),
                train_acc,
                val_acc,
                optimizer.param_groups[0]['lr'],
            )
            prog_bar.set_description(info)
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, order) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs, loss = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, type, loader):
        self._network.eval()
        y_pred, y_true, y_order = [], [], []
        for i, (_, inputs, targets, orders) in enumerate(loader):
            inputs = inputs.to(self._device)
            if type == 'out' or self.args['run_type'] == 'train' or self.args['run_type'] == 'train_bak':
                with torch.no_grad():
                    outputs, loss = self._network(inputs)
                y_order.append(orders.cpu().numpy())
                y_pred.append(outputs.cpu().numpy())
                y_true.append(targets.cpu().numpy())
            else:
                for class_id in self.args['test_class'][self._cur_task]:
                    if class_id in orders.cpu().numpy():
                        with torch.no_grad():
                            outputs, loss = self._network(inputs)
                        y_order.append(orders.cpu().numpy())
                        y_pred.append(outputs.cpu().numpy())
                        y_true.append(targets.cpu().numpy())
        return np.concatenate(y_order), np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval(self, test_loader, out_loader, data_manager):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path)
        self._network.eval()
        orders, _out_k, _pred_k, _labels_k, cnn_accy, nme_accy = self.eval_task(type='test')
        eval_type = 'NME' if nme_accy else 'CNN'
        if eval_type == 'NME':
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
            logging.info("closed-set NME: {}".format(nme_accy["grouped"]))
        else:
            logging.info("closed-set No NME accuracy.")
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
        precision_f, precision_r, recall_f, recall_r, f1_f, f1_r = metrics_others(orders, _pred_k, _labels_k)
        logging.info("closed-set Precision Fake: {}\n".format(precision_f))
        logging.info("closed-set Precision Real: {}\n".format(precision_r))
        logging.info("closed-set Recall Fake: {}\n".format(recall_f))
        logging.info("closed-set Recall Real: {}\n".format(recall_r))
        logging.info("closed-set F1 Fake: {}\n".format(f1_f))
        logging.info("closed-set F1 Real: {}\n".format(f1_r))

    def incremental_eval(self, data_manager):
        self._cur_task = 0
        for i in range(len(self.args["test_class"])):
            print(self._cur_task, self.args["test_class"][i], self.args['trained_path'][i])
            test_dataset = data_manager.get_dataset(
                self.args["test_class"][i], source="test", mode="test", resize_size=self.args["resize_size"]
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
            )
            self._network.to(self._device)
            self._network.train()

            set_random(self.args['seed'])
            self._eval_only(self.args['trained_path'][i])

            self._cur_task += 1

    def _eval_only(self, path):
        set_random(self.args['seed'])
        load_model(self._network, self.args, self.best_model_path, trained_path=path, have_centers=False)
        self._network.eval()
        orders, _out_k, _pred_k, _labels_k, cnn_accy, nme_accy = self.eval_task(type='test')
        eval_type = 'NME' if nme_accy else 'CNN'
        if eval_type == 'NME':
            logging.info("closed-set CNN: {}".format(cnn_accy["grouped"]))
            logging.info("closed-set NME: {}".format(nme_accy["grouped"]))
        else:
            logging.info("closed-set No NME accuracy.")
            logging.info("closed-set CNN: {}\n".format(cnn_accy["grouped"]))
        precision_f, precision_r, recall_f, recall_r, f1_f, f1_r = metrics_others(orders, _pred_k, _labels_k)
        logging.info("closed-set Precision Fake: {}\n".format(precision_f))
        logging.info("closed-set Precision Real: {}\n".format(precision_r))
        logging.info("closed-set Recall Fake: {}\n".format(recall_f))
        logging.info("closed-set Recall Real: {}\n".format(recall_r))
        logging.info("closed-set F1 Fake: {}\n".format(f1_f))
        logging.info("closed-set F1 Real: {}\n".format(f1_r))

    def each_auc(self, prediction, labels, orders, increment=1):
        all_auc = {}
        all_auc["total"] = np.around(
            roc_auc_score(labels, prediction) * 100, decimals=2
        )
        # Grouped accuracy
        for class_id in range(0, np.max(orders) + 1, increment):
            label = "{}".format(
                str(class_id).rjust(2, "0")
            )
            idxes = np.where(orders == class_id)
            if len(idxes[0]) == 0:
                all_auc[label] = 'nan'
                continue
            all_auc[label] = np.around(
                roc_auc_score(labels[idxes], prediction[idxes]) * 100, decimals=2
            )
        return all_auc

    @property
    def feature_dim(self):
        return self._network.head.in_features

    def _construct_random_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            set_random(self.args['seed'])
            logging.info("Constructing exemplars for class {}".format(class_idx))
            real_data, fake_data = [], []
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                resize_size=self.args["resize_size"],
                ret_data=True
            )
            for i in range(len(data)):
                if 'real' in data[i]:
                    real_data.append(data[i])
                else:
                    fake_data.append(data[i])
            real_data = np.array(real_data)
            fake_data = np.array(fake_data)
            selected_exemplars_real = np.array(random.sample(list(real_data), m//2))
            exemplar_targets_real = np.full(selected_exemplars_real.shape[0], class_idx)

            selected_exemplars_fake = np.array(random.sample(list(fake_data), m // 2))
            exemplar_targets_fake = np.full(selected_exemplars_fake.shape[0], class_idx)

            self._real_data_memory = (
                np.concatenate((self._real_data_memory, selected_exemplars_real))
                if len(self._real_data_memory) != 0
                else selected_exemplars_real
            )
            self._real_targets_memory = (
                np.concatenate((self._real_targets_memory, exemplar_targets_real))
                if len(self._real_targets_memory) != 0
                else exemplar_targets_real
            )

            self._fake_data_memory = (
                np.concatenate((self._fake_data_memory, selected_exemplars_fake))
                if len(self._fake_data_memory) != 0
                else selected_exemplars_fake
            )
            self._fake_targets_memory = (
                np.concatenate((self._fake_targets_memory, exemplar_targets_fake))
                if len(self._fake_targets_memory) != 0
                else exemplar_targets_fake
            )

    def _reduce_random_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))

        real_dummy_data, real_dummy_targets = copy.deepcopy(self._real_data_memory), copy.deepcopy(
            self._real_targets_memory
        )
        fake_dummy_data, fake_dummy_targets = copy.deepcopy(self._fake_data_memory), copy.deepcopy(
            self._fake_targets_memory
        )
        self._class_means = np.zeros((2, self._total_classes, self.feature_dim))
        self._real_data_memory, self._real_targets_memory = self._reduce_random_process(real_dummy_data, real_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 0)
        self._fake_data_memory, self._fake_targets_memory = self._reduce_random_process(fake_dummy_data, fake_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 1)

    def _reduce_random_process(self, dummy_data, dummy_targets, data_memory, targets_memory, data_manager, m, label):
        for class_idx in range(self._known_classes):
            set_random(self.args['seed'])
            logging.info("Reducing exemplars for label {} data class {}".format(label, class_idx))
            mask = np.where(dummy_targets == class_idx)[0]

            dd, dt = np.array(random.sample(list(dummy_data[mask]), m)), np.array(random.sample(list(dummy_targets[mask]), m))

            data_memory = (
                np.concatenate((data_memory, dd))
                if len(data_memory) != 0
                else dd
            )
            targets_memory = (
                np.concatenate((targets_memory, dt))
                if len(targets_memory) != 0
                else dt
            )
        return data_memory, targets_memory

    def build_rehearsal_memory(self, data_manager, per_class):
        # self._reduce_exemplar(data_manager, per_class)
        # self._construct_exemplar(data_manager, per_class)
        self._reduce_random_exemplar(data_manager, per_class)
        self._construct_random_exemplar(data_manager, per_class)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        real_dummy_data, real_dummy_targets = copy.deepcopy(self._real_data_memory), copy.deepcopy(
            self._real_targets_memory
        )
        fake_dummy_data, fake_dummy_targets = copy.deepcopy(self._fake_data_memory), copy.deepcopy(
            self._fake_targets_memory
        )
        self._class_means = np.zeros((2, self._total_classes, self.feature_dim))
        self._real_data_memory, self._real_targets_memory = self._reduce_process(real_dummy_data, real_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 0)
        self._fake_data_memory, self._fake_targets_memory = self._reduce_process(fake_dummy_data, fake_dummy_targets,
                                                                                 np.array([]), np.array([]),
                                                                                 data_manager, m // 2, 1)

    def _extract_vectors(self, loader):
        real_vectors, fake_vectors, targets, orders = [], [], [], []
        for _, _inputs, _targets, _orders in loader:
            _targets = _targets.numpy()
            _orders = _orders.numpy()
            _vectors = tensor2numpy(
                self._network.forward_features(_inputs.to(self._device))['features'][:, 0].view(_inputs.size()[0], -1)
            )
            real_vectors.append(_vectors[np.where(_targets == 0)])
            fake_vectors.append(_vectors[np.where(_targets == 1)])
            orders.append(_orders)
            targets.append(_targets)
        if len(real_vectors) == 0:
            return np.concatenate(orders), 0, np.concatenate(fake_vectors), np.concatenate(
                targets)
        elif len(fake_vectors) == 0:
            return np.concatenate(orders), np.concatenate(real_vectors), 0, np.concatenate(targets)
        else:
            return np.concatenate(orders), np.concatenate(real_vectors), np.concatenate(fake_vectors), np.concatenate(
                targets)

    def _reduce_process(self, dummy_data, dummy_targets, data_memory, targets_memory, data_manager, m, label):
        for class_idx in range(self._known_classes):
            set_random(self.args['seed'])
            logging.info("Reducing exemplars for label {} data class {}".format(label, class_idx))
            mask = np.where(dummy_targets == class_idx)[0]

            # trsf = transforms.Compose([transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
            # dataset = DummyDataset(None, np.array(dummy_data[mask]), np.array(dummy_targets[mask]), self.args["resize_size"], trsf)
            # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            # if label:
            #     _, _, vectors, _ = self._extract_vectors(loader)
            # else:
            #     _, vectors, _, _ = self._extract_vectors(loader)
            # index, _ = FPS(vectors, m)
            # dd = dummy_data[mask][index]
            # dt = dummy_targets[mask][:m]

            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

            data_memory = (
                np.concatenate((data_memory, dd))
                if len(data_memory) != 0
                else dd
            )
            targets_memory = (
                np.concatenate((targets_memory, dt))
                if len(targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt), resize_size=self.args["resize_size"]
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            # _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            vectors = real_vectors if not label else fake_vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[label, class_idx, :] = mean
        return data_memory, targets_memory

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            set_random(self.args['seed'])
            logging.info("Constructing exemplars for class {}".format(class_idx))
            real_data, fake_data = [], []
            if self._cur_task == 0:
                shot = None
            else:
                shot = self.args["shot"]
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                resize_size=self.args["resize_size"],
                ret_data=True,
                shot=shot
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            )
            _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
            for i in range(len(data)):
                if 'real' in data[i]:
                    real_data.append(data[i])
                else:
                    fake_data.append(data[i])
            real_data = np.array(real_data)
            fake_data = np.array(fake_data)

            self._construct_process(class_idx, m // 2, real_vectors, data_manager, real_data, 0)
            self._construct_process(class_idx, m // 2, fake_vectors, data_manager, fake_data, 1)

    def _construct_process(self, class_idx, per_class, vectors, data_manager, data, label):
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # index, selected_vectors = FPS(vectors, per_class)
        # selected_exemplars = data[index]

        class_mean = np.mean(vectors, axis=0)

        # Select
        selected_exemplars = []
        exemplar_vectors = []  # [n, feature_dim]
        for k in range(1, per_class + 1):
            S = np.sum(
                exemplar_vectors, axis=0
            )  # [feature_dim] sum of selected exemplars vectors
            mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
            if i >= len(data):
                i = random.randint(0, len(data) - 1)
            selected_exemplars.append(
                np.array(data[i])
            )  # New object to avoid passing by inference
            exemplar_vectors.append(
                np.array(vectors[i])
            )  # New object to avoid passing by inference

            vectors = np.delete(
                vectors, i, axis=0
            )  # Remove it to avoid duplicative selection
            data = np.delete(
                data, i, axis=0
            )  # Remove it to avoid duplicative selection

            if len(vectors) == 0:
                break

        selected_exemplars = np.array(selected_exemplars)
        # exemplar_targets = np.full(m, class_idx)
        exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
        set_random(self.args['seed'])
        if not label:
            self._real_data_memory = (
                np.concatenate((self._real_data_memory, selected_exemplars))
                if len(self._real_data_memory) != 0
                else selected_exemplars
            )
            self._real_targets_memory = (
                np.concatenate((self._real_targets_memory, exemplar_targets))
                if len(self._real_targets_memory) != 0
                else exemplar_targets
            )
        else:
            self._fake_data_memory = (
                np.concatenate((self._fake_data_memory, selected_exemplars))
                if len(self._fake_data_memory) != 0
                else selected_exemplars
            )
            self._fake_targets_memory = (
                np.concatenate((self._fake_targets_memory, exemplar_targets))
                if len(self._fake_targets_memory) != 0
                else exemplar_targets
            )

        # Exemplar mean
        idx_dataset = data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            resize_size=self.args["resize_size"],
            appendent=(selected_exemplars, exemplar_targets),
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        _, real_vectors, fake_vectors, _ = self._extract_vectors(idx_loader)
        vectors = real_vectors if not label else fake_vectors
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)

        self._class_means[label, class_idx, :] = mean


def _FD_loss(student_feature, teacher_feature):
    return torch.nn.functional.mse_loss(F.normalize(student_feature, dim=1), F.normalize(teacher_feature, dim=1),
                                        reduction='mean')


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _JD_loss(logit1, logit2, T):
    prob1 = F.softmax(logit1 / T, dim=1)
    prob2 = F.softmax(logit2 / T, dim=1)

    mean = 0.5 * (prob1 + prob2)

    return 0.5 * (F.kl_div(F.log_softmax(logit1 / T, dim=1), mean, reduction='batchmean') + F.kl_div(
        F.log_softmax(logit2 / T, dim=1), mean, reduction='batchmean'))


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


