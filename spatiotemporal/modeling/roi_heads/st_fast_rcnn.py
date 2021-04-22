from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs


class StFastRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self, box2box_transform, ref_pred_class_logits, st_pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            ref_pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances in the reference (current) frame.
                Each row corresponds to a predicted object instance.
                If None, current frame classification is disabled.
            st_pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances using spatio-temporal information.
                Each row corresponds to a predicted object instance.
                If None, spatio-temporal frame classification is disabled.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """

        # We use st_pred_class_logits here. It means we only use spatio-temporal logits in training since we are not
        # overriding loss calculation methods in FastRCNNOutputs.
        super(StFastRCNNOutputs, self).__init__(
            box2box_transform, 
            st_pred_class_logits,
            pred_proposal_deltas, 
            proposals, smooth_l1_beta
        )
        self.ref_pred_class_logits = ref_pred_class_logits

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        if self.pred_class_logits is not None and self.ref_pred_class_logits is not None:
            st_probs = F.softmax(self.pred_class_logits, dim=-1)  # st_pred_class_logits
            ref_probs = F.softmax(self.ref_pred_class_logits, dim=-1)
            probs = ref_probs + st_probs * (1 - ref_probs)
        
        elif self.pred_class_logits is not None:
            probs = F.softmax(self.pred_class_logits, dim=-1)  # st_pred_class_logits

        elif self.ref_pred_class_logits is not None:
            probs = F.softmax(self.ref_pred_class_logits, dim=-1)  # st_pred_class_logits

        else:
            assert False, "We need at least one classification output."

        return probs.split(self.num_preds_per_image, dim=0)
