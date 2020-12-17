import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = 'sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # TODO: Target has only 1 object per grid cell
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., (self.C + 1):(self.C + 5)], target[..., (self.C + 1):(self.C + 5)])
        iou_b2 = intersection_over_union(predictions[..., (self.C + 6):(self.C + 10)], target[..., (self.C + 1):(self.C + 5)])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        ious_maxes, bestbox = torch.max(ious, dim = 0) # bestbox can be either 0 or 1
        exists_box = target[..., self.C].unsqueeze(3) # Identity of object existance for grid cells
        # print('bestbox: ', bestbox.shape)
        # ======================== #
        # FOR BOX COORDINATES LOSS #
        # ======================== #
        box_predictions = exists_box * (
            bestbox * predictions[..., (self.C + 6):(self.C + 10)]
            + (1 - bestbox) * predictions[..., (self.C + 1):(self.C + 5)]
        )
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )  # TODO: is there case when w and h are negative?

        box_targets = exists_box * target[..., (self.C + 1):(self.C + 5)]
        box_targets[..., 2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # =============== #
        # FOR OBJECT LOSS #
        # =============== #
        pred_box = (
                bestbox * predictions[..., (self.C + 5):(self.C + 6)] + (1 - bestbox) * predictions[..., self.C:(self.C + 1)]
        )

        # N * S * S
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:(self.C + 1)])
        )

        # ================== #
        # FOR NO OBJECT LOSS #
        # ================== #
        # (N, S, S, 1) -> (N, S * S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[...,self.C:(self.C + 1)], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C + 1)], start_dim = 1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., (self.C + 5):(self.C + 6)], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., self.C:(self.C + 1)], start_dim = 1)
        )

        # ================== #
        # FOR CLASS LOSS #
        # ================== #
        # (N, S, S, 20) -> (N * S * S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim = -2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

def test(S = 7, B = 2, C = 20):
    target = torch.ones((13, S,  S,  (C + 5 * B)))
    predictions = torch.randn((13, S * S * (C + 5 * B)))
    print(target.shape, predictions.shape)
    loss_func = YoloLoss(S = S, B = B, C = C)
    loss = loss_func(predictions, target)
    print(loss)

if __name__=='__main__':
	test()