# Based on https://github.com/Scalsol/mega.pytorch/blob/master/mega_core/modeling/roi_heads/box_head/roi_box_feature_extractors.py
# and https://github.com/facebookresearch/detectron2/blob/v0.1.1/detectron2/modeling/roi_heads/box_head.py
import math

import numpy as np
import torch
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY

class AttentionExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(AttentionExtractor, self).__init__()

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0):
        """
        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        if position_embedding is not None:
            position_feat_1 = F.relu(self.Wgs[index](position_embedding))
            # aff_weight, [num_rois, group, num_nongt_rois, 1]
            aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, group, num_nongt_rois]
            aff_weight = aff_weight.squeeze(3)
        
        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        if position_embedding is not None:
            weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        else:
            weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)
        mean_weight = torch.mean(aff_softmax, dim=1)

        return output, mean_weight

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)

        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)

        # TODO: might not be necessary with new PyTorch versions: https://github.com/pytorch/pytorch/issues/29992
        position_embedding = position_embedding.contiguous()

        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding


######################################################################
@ROI_BOX_HEAD_REGISTRY.register()
class AttentionFCHead(AttentionExtractor):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg, input_shape)


        # TODO: move to layers.py?
        def make_fc(dim_in, hidden_dim, use_gn=False):
            '''
                Caffe2 implementation uses XavierFill, which in fact
                corresponds to kaiming_uniform_ in PyTorch
            '''
            assert not use_gn
            # if use_gn:
            #     fc = nn.Linear(dim_in, hidden_dim, bias=False)
            #     nn.init.kaiming_uniform_(fc.weight, a=1)
            #     return nn.Sequential(fc, group_norm(hidden_dim))
            fc = nn.Linear(dim_in, hidden_dim)
            nn.init.kaiming_uniform_(fc.weight, a=1)
            nn.init.constant_(fc.bias, 0)
            return fc

        # fmt: off
        fc_dim                  = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm                    = cfg.MODEL.ROI_BOX_HEAD.NORM

        self.embed_dim          = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.ATTENTION.EMBED_DIM
        self.groups             = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.ATTENTION.GROUP
        self.feat_dim           = fc_dim
        self.base_stage         = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.ATTENTION.STAGE
        self.advanced_stage     = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.ATTENTION.ADVANCED_STAGE
        self.base_num           = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.REF_POST_NMS_TOP_N
        self.advanced_num       = int(self.base_num * cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.RDN_RATIO)
        self.location_free      = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.ATTENTION.LOCATION_FREE
        # fmt: on

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        input_size = np.prod((input_shape.channels, input_shape.height, input_shape.width))


        fcs, Wgs, Wqs, Wks, Wvs = [], [], [], [], []
        for i in range(self.base_stage + self.advanced_stage + 1):
            r_size = input_size if i == 0 else fc_dim

            if i == self.base_stage and self.advanced_stage == 0:
                break

            if i != self.base_stage + self.advanced_stage:
                fcs.append(make_fc(r_size, fc_dim))
                self._output_size = fc_dim
            Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
            Wqs.append(make_fc(self.feat_dim, self.feat_dim))
            Wks.append(make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0, groups=self.groups))
            for l in [Wgs[i], Wvs[i]]:
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)
        self.fcs = nn.ModuleList(fcs)
        self.Wgs = nn.ModuleList(Wgs)
        self.Wqs = nn.ModuleList(Wqs)
        self.Wks = nn.ModuleList(Wks)
        self.Wvs = nn.ModuleList(Wvs)


    def forward(self, x, proposals, long_term_feature_buffer, long_term_roi_buffer, pre_calculate=False):
        if self.training:
            return self._forward_train(x, proposals, long_term_feature_buffer, long_term_roi_buffer)
        else:
            return self._forward_test(x, proposals, long_term_feature_buffer, long_term_roi_buffer)


    def _forward_test(self, x, proposals, long_term_feature_buffer, long_term_roi_buffer):

        assert len(proposals) == 1
        rois_cur = proposals[0].tensor
        rois_ref = torch.cat(list(long_term_roi_buffer) + [rois_cur[:self.base_num]])
        x = x.flatten(start_dim=1)
        
        if not self.location_free:
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        else:
            position_embedding = None

        for i in range(self.base_stage):
            x = F.relu(self.fcs[i](x))
            if i == 0:
                x_out = x[:self.base_num]
                x_refs = torch.cat(list(long_term_feature_buffer) + [x[:self.base_num]])
            attention, weights = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            weights = weights.reshape(weights.shape[0], len(long_term_feature_buffer)+1,-1)

            x = x + attention

        if self.advanced_stage > 0:
            x_refs_adv = torch.cat([x[:self.advanced_num] for x in torch.split(x_refs, self.base_num, dim=0)], dim=0)
            rois_ref_adv = torch.cat([x[:self.advanced_num] for x in torch.split(rois_ref, self.base_num, dim=0)], dim=0)

            if not self.location_free:
                position_embedding_adv = torch.cat([x[..., :self.advanced_num] for x in torch.split(position_embedding, self.base_num, dim=-1)], dim=-1)
                position_embedding = self.cal_position_embedding(rois_ref_adv, rois_ref)
            else:
                position_embedding_adv = None
                position_embedding = None

            for i in range(self.advanced_stage):
                attention, _ = self.attention_module_multi_head(x_refs_adv, x_refs, position_embedding,
                                                             feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                             index=i + self.base_stage)
                x_refs_adv = x_refs_adv + attention
                x_refs_adv = F.relu(self.fcs[i + self.base_stage](x_refs_adv))

            attention, _ = self.attention_module_multi_head(x, x_refs_adv, position_embedding_adv,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=self.base_stage + self.advanced_stage)
            x = x + attention

        return x, x_out, weights


    def _forward_train(self, x, proposals, long_term_feature_buffer, long_term_roi_buffer):

        assert len(proposals) == 1
        rois_cur = proposals[0].tensor
        rois_ref = torch.cat(list(long_term_roi_buffer))

        if not self.location_free:
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        else:
            position_embedding = None

        long_term_feature_buffer = [f.flatten(start_dim=1) for f in long_term_feature_buffer]
        x = x.flatten(start_dim=1)

        x_refs = torch.cat(list(long_term_feature_buffer))
        x_refs = F.relu(self.fcs[0](x_refs))

        for i in range(self.base_stage):
            x = F.relu(self.fcs[i](x))
            attention, _ = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            x = x + attention

        if self.advanced_stage > 0:
            x_refs_adv = torch.cat([x[:self.advanced_num] for x in torch.split(x_refs, self.base_num, dim=0)], dim=0)
            rois_ref_adv = torch.cat([x[:self.advanced_num] for x in torch.split(rois_ref, self.base_num, dim=0)], dim=0)

            if not self.location_free:
                position_embedding_adv = torch.cat([x[..., :self.advanced_num] for x in torch.split(position_embedding, self.base_num, dim=-1)], dim=-1)
                position_embedding = self.cal_position_embedding(rois_ref_adv, rois_ref)
            else:
                position_embedding_adv = None
                position_embedding = None

            for i in range(self.advanced_stage):
                attention, _ = self.attention_module_multi_head(x_refs_adv, x_refs, position_embedding,
                                                             feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                             index=i + self.base_stage)
                x_refs_adv = x_refs_adv + attention
                x_refs_adv = F.relu(self.fcs[i + self.base_stage](x_refs_adv))

            attention, _ = self.attention_module_multi_head(x, x_refs_adv, position_embedding_adv,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=self.base_stage + self.advanced_stage)
            x = x + attention

        return x, None, None


    @property
    def output_size(self):
        return self._output_size


def build_st_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.SPATIOTEMPORAL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)