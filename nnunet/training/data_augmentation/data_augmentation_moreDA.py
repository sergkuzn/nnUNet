#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter, MultiThreadedAugmenterFixMatch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True, regions=None,
                            use_nondetMultiThreadedAugmenter: bool = False):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    if params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get("do_additive_brightness"):
        tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                 params.get("additive_brightness_sigma"),
                                                 True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                 p_per_channel=params.get("additive_brightness_p_per_channel")))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror") or params.get("mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                "cascade_do_cascade_augmentations"):
            if params.get("cascade_random_binary_transform_p") > 0:
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size"),
                    p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
            if params.get("cascade_remove_conn_comp_p") > 0:
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get("num_cached_per_thread"), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    return batchgenerator_train, batchgenerator_val


def get_moreDA_augmentation_weak(dataloader_train, dataloader_val, patch_size,
                                 params_weak=default_3D_augmentation_params,
                                 border_val_seg=-1,
                                 seeds_train=None, seeds_val=None, order_seg=1, order_data=3,
                                 deep_supervision_scales=None,
                                 soft_ds=False,
                                 classes=None, pin_memory=True, regions=None,
                                 use_nondetMultiThreadedAugmenter: bool = False):
    """
    Same as get_moreDA_augmentation but with weak params.
    """
    assert params_weak.get('mirror') is None, "old version of params, use new keyword do_mirror"

    transforms_weak = []

    if params_weak.get("selected_data_channels") is not None:
        transforms_weak.append(DataChannelSelectionTransform(params_weak.get("selected_data_channels")))

    if params_weak.get("selected_seg_channels") is not None:
        transforms_weak.append(SegChannelSelectionTransform(params_weak.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params_weak.get("dummy_2D") is not None and params_weak.get("dummy_2D"):
        ignore_axes = (0,)
        transforms_weak.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    transforms_weak.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params_weak.get("do_elastic"), alpha=params_weak.get("elastic_deform_alpha"),
        sigma=params_weak.get("elastic_deform_sigma"),
        do_rotation=params_weak.get("do_rotation"), angle_x=params_weak.get("rotation_x"),
        angle_y=params_weak.get("rotation_y"),
        angle_z=params_weak.get("rotation_z"), p_rot_per_axis=params_weak.get("rotation_p_per_axis"),
        do_scale=params_weak.get("do_scaling"), scale=params_weak.get("scale_range"),
        border_mode_data=params_weak.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params_weak.get("random_crop"), p_el_per_sample=params_weak.get("p_eldef"),
        p_scale_per_sample=params_weak.get("p_scale"), p_rot_per_sample=params_weak.get("p_rot"),
        independent_scale_for_each_axis=params_weak.get("independent_scale_factor_for_each_axis")
    ))

    if params_weak.get("dummy_2D"):
        transforms_weak.append(Convert2DTo3DTransform())

    if params_weak.get("do_mirror") or params_weak.get("mirror"):
        transforms_weak.append(MirrorTransform(params_weak.get("mirror_axes")))

    if params_weak.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params_weak.get("mask_was_used_for_normalization")
        transforms_weak.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    transforms_weak.append(RemoveLabelTransform(-1, 0))

    transforms_weak.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        transforms_weak.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            transforms_weak.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            transforms_weak.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                output_key='target'))

    transforms_weak.append(NumpyToTensor(['data', 'target'], 'float'))
    transforms_weak = Compose(transforms_weak)

    if use_nondetMultiThreadedAugmenter:
        raise NotImplementedError()
        # if NonDetMultiThreadedAugmenter is None:
        #     raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        # batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, transforms_weak, params.get('num_threads'),
        #                                                     params.get("num_cached_per_thread"), seeds=seeds_train,
        #                                                     pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, transforms_weak,
                                                      params_weak.get('num_threads'),
                                                      params_weak.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, transforms_weak)
    # import IPython;IPython.embed()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params_weak.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params_weak.get("selected_data_channels")))
    if params_weak.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params_weak.get("selected_seg_channels")))

    if params_weak.get("move_last_seg_chanel_to_data") is not None and params_weak.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params_weak.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params_weak.get('num_threads') // 2, 1),
                                                          params_weak.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params_weak.get('num_threads') // 2, 1),
                                                    params_weak.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    return batchgenerator_train, batchgenerator_val


def get_moreDA_augmentation_unlabeled(dataloader_train, patch_size_strong, patch_size_weak,
                                      params_strong=default_3D_augmentation_params,
                                      params_weak=None,
                                      border_val_seg=-1,
                                      seeds_train=None, seeds_val=None, order_seg=1, order_data=3,
                                      deep_supervision_scales=None,
                                      soft_ds=False,
                                      classes=None, pin_memory=True, regions=None,
                                      use_nondetMultiThreadedAugmenter: bool = False):
    assert params_strong.get('mirror') is None, "old version of params, use new keyword do_mirror"

    transforms_strong = []

    if params_strong.get("selected_data_channels") is not None:
        transforms_strong.append(DataChannelSelectionTransform(params_strong.get("selected_data_channels")))

    if params_strong.get("selected_seg_channels") is not None:
        transforms_strong.append(SegChannelSelectionTransform(params_strong.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params_strong.get("dummy_2D") is not None and params_strong.get("dummy_2D"):
        ignore_axes = (0,)
        transforms_strong.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size_strong[1:]
    else:
        patch_size_spatial = patch_size_strong
        ignore_axes = None

    transforms_strong.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params_strong.get("do_elastic"), alpha=params_strong.get("elastic_deform_alpha"),
        sigma=params_strong.get("elastic_deform_sigma"),
        do_rotation=params_strong.get("do_rotation"), angle_x=params_strong.get("rotation_x"),
        angle_y=params_strong.get("rotation_y"),
        angle_z=params_strong.get("rotation_z"), p_rot_per_axis=params_strong.get("rotation_p_per_axis"),
        do_scale=params_strong.get("do_scaling"), scale=params_strong.get("scale_range"),
        border_mode_data=params_strong.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params_strong.get("random_crop"), p_el_per_sample=params_strong.get("p_eldef"),
        p_scale_per_sample=params_strong.get("p_scale"), p_rot_per_sample=params_strong.get("p_rot"),
        independent_scale_for_each_axis=params_strong.get("independent_scale_factor_for_each_axis")
    ))

    if params_strong.get("dummy_2D"):
        transforms_strong.append(Convert2DTo3DTransform())

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    transforms_strong.append(GaussianNoiseTransform(p_per_sample=0.1))
    transforms_strong.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
    transforms_strong.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params_strong.get("do_additive_brightness"):
        transforms_strong.append(BrightnessTransform(params_strong.get("additive_brightness_mu"),
                                                     params_strong.get("additive_brightness_sigma"),
                                                     True,
                                                     p_per_sample=params_strong.get("additive_brightness_p_per_sample"),
                                                     p_per_channel=params_strong.get(
                                                         "additive_brightness_p_per_channel")))

    transforms_strong.append(ContrastAugmentationTransform(p_per_sample=0.15))
    transforms_strong.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
    transforms_strong.append(
        GammaTransform(params_strong.get("gamma_range"), True, True,
                       retain_stats=params_strong.get("gamma_retain_stats"),
                       p_per_sample=0.1))  # inverted gamma

    if params_strong.get("do_gamma"):
        transforms_strong.append(
            GammaTransform(params_strong.get("gamma_range"), False, True,
                           retain_stats=params_strong.get("gamma_retain_stats"),
                           p_per_sample=params_strong["p_gamma"]))

    if params_strong.get("do_mirror") or params_strong.get("mirror"):
        transforms_strong.append(MirrorTransform(params_strong.get("mirror_axes")))

    if params_strong.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params_strong.get("mask_was_used_for_normalization")
        transforms_strong.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    transforms_strong.append(RemoveLabelTransform(-1, 0))

    if params_strong.get("move_last_seg_chanel_to_data") is not None and params_strong.get(
            "move_last_seg_chanel_to_data"):
        transforms_strong.append(MoveSegAsOneHotToData(1, params_strong.get("all_segmentation_labels"), 'seg', 'data'))
        if params_strong.get("cascade_do_cascade_augmentations") is not None and params_strong.get(
                "cascade_do_cascade_augmentations"):
            if params_strong.get("cascade_random_binary_transform_p") > 0:
                transforms_strong.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params_strong.get("all_segmentation_labels")), 0)),
                    p_per_sample=params_strong.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params_strong.get("cascade_random_binary_transform_size"),
                    p_per_label=params_strong.get("cascade_random_binary_transform_p_per_label")))
            if params_strong.get("cascade_remove_conn_comp_p") > 0:
                transforms_strong.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(params_strong.get("all_segmentation_labels")), 0)),
                        key="data",
                        p_per_sample=params_strong.get("cascade_remove_conn_comp_p"),
                        fill_with_other_class_p=params_strong.get(
                            "cascade_remove_conn_comp_max_size_percent_threshold"),
                        dont_do_if_covers_more_than_X_percent=params_strong.get(
                            "cascade_remove_conn_comp_fill_with_other_class_p")))

    transforms_strong.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        transforms_strong.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            transforms_strong.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            transforms_strong.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                  output_key='target'))

    transforms_strong.append(NumpyToTensor(['data', 'target'], 'float'))
    transforms_strong = Compose(transforms_strong)

    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, transforms_strong)
    # import IPython;IPython.embed()

    transforms_weak = []

    if params_weak.get("selected_data_channels") is not None:
        transforms_weak.append(DataChannelSelectionTransform(params_weak.get("selected_data_channels")))

    if params_weak.get("selected_seg_channels") is not None:
        transforms_weak.append(SegChannelSelectionTransform(params_weak.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params_weak.get("dummy_2D") is not None and params_weak.get("dummy_2D"):
        ignore_axes = (0,)
        transforms_weak.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size_weak[1:]
    else:
        patch_size_spatial = patch_size_weak
        ignore_axes = None

    transforms_weak.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=params_weak.get("do_elastic"), alpha=params_weak.get("elastic_deform_alpha"),
        sigma=params_weak.get("elastic_deform_sigma"),
        do_rotation=params_weak.get("do_rotation"), angle_x=params_weak.get("rotation_x"),
        angle_y=params_weak.get("rotation_y"),
        angle_z=params_weak.get("rotation_z"), p_rot_per_axis=params_weak.get("rotation_p_per_axis"),
        do_scale=params_weak.get("do_scaling"), scale=params_weak.get("scale_range"),
        border_mode_data=params_weak.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params_weak.get("random_crop"), p_el_per_sample=params_weak.get("p_eldef"),
        p_scale_per_sample=params_weak.get("p_scale"), p_rot_per_sample=params_weak.get("p_rot"),
        independent_scale_for_each_axis=params_weak.get("independent_scale_factor_for_each_axis")
    ))

    if params_weak.get("dummy_2D"):
        transforms_weak.append(Convert2DTo3DTransform())

    if params_weak.get("do_mirror") or params_weak.get("mirror"):
        transforms_weak.append(MirrorTransform(params_weak.get("mirror_axes")))

    if params_weak.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params_weak.get("mask_was_used_for_normalization")
        transforms_weak.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    transforms_weak.append(RemoveLabelTransform(-1, 0))

    transforms_weak.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        transforms_weak.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            transforms_weak.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            transforms_weak.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                output_key='target'))

    transforms_weak.append(NumpyToTensor(['data', 'target'], 'float'))
    transforms_weak = Compose(transforms_weak)

    batchgenerator = MultiThreadedAugmenterFixMatch(dataloader_train, transforms_strong, transforms_weak,
                                                    max(params_strong.get('num_threads') // 2, 1),
                                                    params_strong.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, transforms_weak)

    return batchgenerator
