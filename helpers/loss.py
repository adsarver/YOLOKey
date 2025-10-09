import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MDPIoU=False, feat_h=640, feat_w=640, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
        return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU
    return iou  # IoU

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.m = m
        self.autobalance = autobalance
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if self.autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.BCE_base = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, p, targets):  # predictions, targets
        self.anchors = self.m.anchors
        self.ssi = list(self.m.stride).index(16) if self.autobalance else 0  # stride 16 index
        tcls, tbox, indices = self.build_targets(p, targets)  # targets
        bs = p[0].shape[0]  # batch size
        n_labels = targets.shape[0]  # number of labels
        loss = torch.zeros(3, device=self.device)  # [box, obj, cls] losses

        # Compute all losses
        all_loss = []
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            if n_labels:
                pxy, pwh, pobj, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 2)  # target-subset of predictions

                # Regression
                pbox = torch.cat((pxy.sigmoid() * 1.6 - 0.3, (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]), 2)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(predicted_box, target_box)
                obj_target = iou.detach().clamp(0).type(pi.dtype)  # objectness targets

                all_loss.append([(1.0 - iou) * self.hyp['box'],
                                 self.BCE_base(pobj.squeeze(), torch.ones_like(obj_target)) * self.hyp['obj'],
                                 self.BCE_base(pcls, F.one_hot(tcls[i], self.nc).float()).mean(2) * self.hyp['cls'],
                                 obj_target,
                                 tbox[i][..., 2] > 0.0])  # valid

        # Lowest 3 losses per label
        n_assign = 4  # top n matches
        cat_loss = [torch.cat(x, 1) for x in zip(*all_loss)]
        ij = torch.zeros_like(cat_loss[0]).bool()  # top 3 mask
        sum_loss = cat_loss[0] + cat_loss[2]
        for col in torch.argsort(sum_loss, dim=1).T[:n_assign]:
            # ij[range(n_labels), col] = True
            ij[range(n_labels), col] = cat_loss[4][range(n_labels), col]
        loss[0] = cat_loss[0][ij].mean() * self.nl  # box loss
        loss[2] = cat_loss[2][ij].mean() * self.nl  # cls loss

        # Obj loss
        for i, (h, pi) in enumerate(zip(ij.chunk(self.nl, 1), p)):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # obj
            if n_labels:  # if any labels
                tobj[b[h], gj[h], gi[h]] = all_loss[i][3][h]
            loss[1] += self.BCEobj(pi[:, 4], tobj) * (self.balance[i] * self.hyp['obj'])

        return loss.sum() * bs, loss.detach()  # [box, obj, cls] losses

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)        
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.3  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float()  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # # Matches
                print(t[..., 4:6].shape, self.anchors.shape)
                r = t[..., 4:6] / self.anchors.T # wh ratio
                a = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # a = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[a]  # filter

                # # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) & a
                t = t.repeat((5, 1, 1))
                offsets = torch.zeros_like(gxy)[None] + off[:, None]
                t[..., 4:6][~j] = 0.0  # move unsuitable targets far away
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 2)  # (image, class), grid xy, grid wh
            b, c = bc.long().transpose(0, 2).contiguous()  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.transpose(0, 2).contiguous()  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy - gij, gwh), 2).permute(1, 0, 2).contiguous())  # box
            tcls.append(c)  # class

            # # Unique
            # n1 = torch.cat((b.view(-1, 1), tbox[i].view(-1, 4)), 1).shape[0]
            # n2 = tbox[i].view(-1, 4).unique(dim=0).shape[0]
            # print(f'targets-unique {n1}-{n2} diff={n1-n2}')

        return tcls, tbox, indices
