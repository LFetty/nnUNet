import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDataLoader2D_Regression(nnUNetDataLoader):
    """
    Modified data loader for 2D image-to-image regression tasks.
    Key changes:
    - Uses float32 for targets instead of int16
    - Uses 0 as background padding for targets instead of -1
    """
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)  # float32 instead of int16
        case_properties = []

        for j, current_key in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(current_key)
            case_properties.append(properties)

            # Handle 2D case - select a slice
            if len(data.shape) == 4:  # 3D data (channels, z, x, y)
                # select a class/region first, then a slice where this class is present, then crop to that area
                if not force_fg:
                    if self.has_ignore:
                        selected_class_or_region = self.annotated_classes_key
                    else:
                        selected_class_or_region = None
                else:
                    # filter out all classes that are not present here
                    eligible_classes_or_regions = [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) > 0]

                    # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                    # strange formulation needed to circumvent
                    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                    tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                    if any(tmp):
                        if len(eligible_classes_or_regions) > 1:
                            eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                    selected_class_or_region = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        len(eligible_classes_or_regions) > 0 else None
                        
                if selected_class_or_region is not None:
                    selected_slice = np.random.choice(properties['class_locations'][selected_class_or_region][:, 1])
                else:
                    selected_slice = np.random.choice(len(data[0]))

                data = data[:, selected_slice]
                seg = seg[:, selected_slice]
                if seg_prev is not None:
                    seg_prev = seg_prev[:, selected_slice]

                # the line of death lol
                # this needs to be a separate variable because we could otherwise permanently overwrite
                # properties['class_locations']
                class_locations = {
                    selected_class_or_region: properties['class_locations'][selected_class_or_region][properties['class_locations'][selected_class_or_region][:, 1] == selected_slice][:, (0, 2, 3)]
                } if (selected_class_or_region is not None) else None
                
                shape = data.shape[1:]
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg if selected_class_or_region is not None else None,
                                                   class_locations, overwrite_class=selected_class_or_region)
            else:
                # Already 2D
                shape = data.shape[1:]
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # Use ACVL utils for cropping and padding
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            
            seg_cropped = crop_and_pad_nd(seg, bbox, 0)  # Use 0 instead of -1 for regression
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, 0)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).float()  # float instead of int16
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'properties': case_properties, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3D_Regression(nnUNetDataLoader):
    """
    Modified data loader for 3D image-to-image regression tasks.
    Key changes:
    - Uses float32 for targets instead of int16
    - Uses 0 as background padding for targets instead of -1
    """
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)  # float32 instead of int16
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # Use ACVL utils for cropping and padding
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            
            seg_cropped = crop_and_pad_nd(seg, bbox, 0)  # Use 0 instead of -1 for regression
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, 0)[None]))
            seg_all[j] = seg_cropped

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).float()  # float instead of int16
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'properties': case_properties, 'keys': selected_keys}
        
        return {'data': data_all, 'target': seg_all, 'properties': case_properties, 'keys': selected_keys}