# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherTopN(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_top_k: int = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_top_k: This is the number of closest predictions to match a groundtruth
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_top_k = cost_top_k
        assert (cost_class != 0 or cost_bbox != 0 or cost_giou != 0) and cost_top_k !=0, "all costs cant be 0 and cost_top_k can't be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size, num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)             # [batch_size, num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])  # tgt_ids[M] = tensor of target class labels in a batch
            tgt_bbox = torch.cat([v["boxes"] for v in targets])  # v is each target (1 image) in list of targets (1 batch)

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())  # same size as out_prob[N, num_classes]  (N = batch_size * num_queries)
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())      # same size as out_prob[N, num_classes]
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # (pos - neg) 1st dim is in the order of tgt_ids => [N, M] matrix (M is the number off classes appear in groundtruths)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # euclidean distance matrix[N, M] between each point in out_bbox[N] and each point in tgt_bbox[M]

            # Compute the giou cost between boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))  # return a matrix[N, M] of giou between out_box[N] and tgt_box[M]

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou  # C = [N, M] matrix (N querries, M groundtruths)
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v["boxes"]) for v in targets]  # get a list of the number of boxes for each image in the batch

            # Find the minimum cost point to each groundtruth point (bipartite matching)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            """ This module will find n points with lowest cost to groundtruth, 
            no matter if these n points overlap with the ones in bipartite matching.
            These n points will be pulled toward groundtruth, but less than the bipartite matching ones (lower loss) """
            if (self.cost_top_k > 1):
                """
                # Method1: bipartite matching
                concat = C
                for _ in range(self.cost_top_k - 1):  # stack N cost matrices horizontally (to multiply groundtruths N times)
                    concat = torch.cat((concat, C), 2)
                D = concat
                NUM = num_queries
                while NUM < (sum(sizes) * self.cost_top_k):  # stack N of the matrix above vertically (if number of groundtruth > number of queries)
                    concat = torch.cat((concat, D), 1)
                    NUM += num_queries
                sizes_k = list(np.array(sizes) * self.cost_top_k)
                top_k_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(concat.split(sizes_k, -1))]
                top_k = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in top_k_indices]
                n = 0
                # since cost matrix rows and columns were multiply, we % the original rows and columns to get the right coords
                for i in top_k:  # cost matrix for each target (i) in a batch (top_k)
                    for v in i[1]:           # get the right column (groundtruth coord)
                        if v >= sizes[n]:
                            v %= sizes[n]
                    for v in i[0]:           # get the right row (query coord)
                        if v >= num_queries:
                            v %= num_queries
                    n += 1
              
                """
                # Method2: find top_k queries for each gt
                top_k = []
                for i, c in enumerate(C.split(sizes, -1)):  # split C into a list of tensors c of size: [N, image_num_box]
                    # both c and i will +1 each loop
                    top = min(self.cost_top_k, num_queries)  # pick the smaller value between k and num_queries
                    top_sort = torch.topk(c[i], top, 0, largest=False)  # get the k smallest values in c[i] dimension 0 (querries dim)
                    # print(top_sort)
                    indices_columns = top_sort.indices.flatten(0)  # get the indices tensor and then flatten dim 0
                    # print(indices_columns)
                    indices_row = torch.arange(c[i].size(1))
                    indices_rows = indices_row                                    # concat indices_row "top" times in dim 0
                    for _ in range(top - 1):                                      #
                        indices_rows = torch.cat((indices_rows, indices_row), 0)  #
                    # print(indices_rows)
                    top_k += [(indices_columns, indices_rows)]
                #print(top_k)


            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], top_k


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)


def build_top_k_matcher(args):
    return HungarianMatcherTopN(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou,
                                cost_top_k=args.set_top_K)