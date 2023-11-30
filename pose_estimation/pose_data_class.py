import os
import pickle
import numpy as np
from PIL import Image
import trimesh
from matplotlib.pyplot import get_cmap

import torch

from .utils import back_project

NUM_OBJECTS = 79
cmap = get_cmap('rainbow', NUM_OBJECTS)
COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(NUM_OBJECTS + 3)])
COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
COLOR_PALETTE[-3] = [119, 135, 150]
COLOR_PALETTE[-2] = [176, 194, 216]
COLOR_PALETTE[-1] = [255, 255, 225]

class PoseData:

    """
    scene : 
        - color : RGB image
        - depth : from camera view (convert from mm to m)
        - label : a segmentation image capture form a camera containing the target objects. The segmentations id for objects are from 0 to 78
        - meta : camera parameters, object names, ground_truth poses (comparison labels)
        - key : CUSTOM : level - scene - variant
    meta : 
        - poses_world (list) : 
            The length is NUM_OBJECTS
            A pose is a 4x4 transformation matrix (rotation and translation) for each object in the world frame, or None for non-existing objects.
        - extents (list) : 
            The length is NUM_OBJECTS
            An extent is a(3,) array, representing the size of each object in its canonical frame (without scaling), or None for non-existing objects. The order is xyz.
        - scales (list) : 
            The length is NUM_OBJECTS
            A scale is a (3,) array, representing the scale of each object, or None for non-existing objects
        - object_names(list):
            the object names of interest.
        - extrinsic:
            4x4 transformation matrix, world -> viewer(opencv)
        - intrinsic :
            3x3 matrix, viewer(opencv) -> image
        - object_ids (list) : 
            the object ids of interest.
    """

    INFO_HEADERS = ['object', 'class', 'source', 'location', 'metric', 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z', 'width', 'length', 'height', 'visual_symmetry', 'geometric_symmetry']
    CSV_FILENAME = "objects_v1.csv"
    DATA_FOLDERNAME = "v2.2"

    def __init__(self, data_path, models_path, split_processed_data=None) -> None:

        self.data_path = data_path
        self.models_path = models_path

        if split_processed_data is not None:
            self.objects, self.data, self.nested_data = split_processed_data
        else:
            self.objects = self.organize_objects(os.path.join(data_path, self.CSV_FILENAME))
            self.data, self.nested_data = self.organize_data(os.path.join(data_path, self.DATA_FOLDERNAME))

        self.keylist = list(self.data.keys())

    def organize_objects(self, objects_path):
        object_lists = np.genfromtxt(objects_path, delimiter=',', skip_header=1, dtype=None, encoding="utf-8")
        objects = []
        for object_list in object_lists:
            objects_dict = {header : value for header, value in zip(self.INFO_HEADERS, object_list)}
            objects.append(objects_dict)
        return objects

    def get_mesh(self, object_id):
        object_info = self.objects[object_id]
        location = object_info["location"].split("/")[-1]
        visual_dae_path = os.path.join(self.models_path, location, "visual_meshes", "visual.dae")
        mesh = trimesh.load(visual_dae_path, force="mesh")
        return mesh

    def get_info(self, object_id):
        return self.objects[object_id]

    def organize_data(self, data_path):
        data = {}
        nested_data = {}
        components = ["color", "depth", "label", "meta"]

        def get_component(filename):
            for component in components:
                if component in filename:
                    return component
            else:
                raise KeyError(f"{filename}")

        for filename in os.listdir(data_path):

            filepath = os.path.join(data_path, filename)
            component = get_component(filename)
            level, scene, variant = [int(idx) for idx in filename.split("_")[0].split("-")]
            key = (level, scene, variant)

            if key not in data:
                data[key] = {"key" : key}

            if level not in nested_data:
                nested_data[level] = {}
            if scene not in nested_data[level]:
                nested_data[level][scene] = {}
            if variant not in nested_data[level][scene]:
                nested_data[level][scene][variant] = {"key" : key}

            if component == "meta" :
                with open(filepath, "rb") as f:
                    entry = pickle.load(f)
            else:
                # Normlize condition
                normalizer = 255 if component == "color" \
                        else 1000 if component == "depth" \
                        else None
                # Closure over variable
                def closure(filepath, normalizer):
                    # Generate PNG
                    def generator():
                        if normalizer is not None:
                            return np.array(Image.open(filepath)) / normalizer
                        return np.array(Image.open(filepath))
                    return generator
                entry = closure(filepath, normalizer)

            # Save memory by making these point to the same object
            data[key][component] = nested_data[level][scene][variant][component] = entry

        return data, nested_data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.keylist)

    def __call__(self, idx):
        # Get from global, flattened index
        return self.data[self.keylist[idx]]

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = (1,)
        if len(indices) == 3:
            return self.data[indices]
        value = self.nested_data
        for i in indices:
            value = value[i]
        return value

    def level_split(self, selected_levels):
        
        if not isinstance(selected_levels, (tuple, list)):
            selected_levels = [selected_levels]

        # Create splits
        split_data = {}
        split_nested_data = {}
        for key in self.data.keys():
            level, scene, variant = key

            if level not in selected_levels:
                continue

            split_data[key] = self.data[key]

            if level not in split_nested_data:
                split_nested_data[level] = {}
            if scene not in split_nested_data[level]:
                split_nested_data[level][scene] = {}
            if variant not in split_nested_data[level][scene]:
                split_nested_data[level][scene][variant] = self.data[key]

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data))

    def txt_split(self, split_txt_path):

        # Load split txt file
        string_indices = np.loadtxt(split_txt_path, dtype=str)

        # Convert to Tuple
        tuple_indices = []
        for string in string_indices:
            indices = [int(idx) for idx in string.split("-")]
            tuple_indices.append(tuple(indices))

        # Create splits
        split_data = {}
        split_nested_data = {}
        for key in tuple_indices:
            level, scene, variant = key

            split_data[key] = self.data[key]

            if level not in split_nested_data:
                split_nested_data[level] = {}
            if scene not in split_nested_data[level]:
                split_nested_data[level][scene] = {}
            if variant not in split_nested_data[level][scene]:
                split_nested_data[level][scene][variant] = self.data[key]

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data))

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, models_path, levels=None, split=None, mesh_samples=None):

        pose_data = PoseData(data_path, models_path)

        if split is not None:
            pose_data = pose_data.txt_split(split)
        if levels is not None:
            pose_data = pose_data.level_split(levels)

        self.object_infos = []
        self.source_meshes = []
        for i in range(len(pose_data.objects)):
            self.object_info.append(pose_data.get_info(i))
            self.source_meshes.append(pose_data.get_mesh(i))
    
        self.object_ids = []
        self.source_points = [] # From Models ; Baseline Objects
        self.target_points = [] # Back Projected ; Off-Pose
        self.target_poses = [] # Ground Truth Poses of Target Points
        self.keys = []
        for i, key in enumerate(pose_data.keys()):

            l, s, v = key

            scene = pose_data[key]

            rgb = scene["color"]()
            depth = scene["depth"]()
            label = scene["label"]()
            meta = scene["meta"]

            back_projection = back_project(depth, meta)

            world_frames = [None] * 79

            object_ids = [object_id for object_id in np.unique(label) if object_id < 79]

            for object_id in object_ids:
                
                indices = np.where(label == object_id)
                target_pcd = back_projection[indices]

                sample_count = len(target_pcd) if mesh_samples is None else mesh_samples
                source_pcd, faces = trimesh.sample.sample_surface(self.source_meshes[object_id], sample_count)

                try:
                    target_pose = meta["poses_world"][object_id][:4, :] # 4x4 -> 3x4
                except KeyError:
                    target_pose = None

                self.object_ids.append(object_id)
                self.source_points.append(torch.tensor(source_pcd).float())
                self.target_points.append(torch.tensor(target_pcd).float())
                self.target_poses.append(torch.tensor(target_pose).float() if target_pose is not None else None)
                self.keys.append((l, s, v))
                
    def __len__(self):
        return len(self.source_points)
    
    def __getitem__(self, i):

        object_id = self.object_ids[i]

        extras = (
            object_id,
            self.object_infos[object_id], # info
            self.source_meshes[object_id], # mesh
            self.keys[i], # scene key (level, scene, variant)
        )

        source_pcd = self.source_points[i]
        target_pcd = self.target_points[i]
        target_pose = self.target_poses[i]

        return source_pcd, target_pcd, target_pose, extras
