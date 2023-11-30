import os
import shutil
import pickle
import numpy as np
from PIL import Image
import trimesh
from matplotlib.pyplot import get_cmap

import torch

from .utils import back_project, crop_image_using_segmentation

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

    def __init__(self, data_path, models_path, object_cache=True, split_processed_data=None) -> None:

        self.data_path = data_path
        self.models_path = models_path

        if split_processed_data is not None:
            self.objects, self.data, self.nested_data, self.object_cache = split_processed_data
        else:
            self.objects = self.organize_objects(os.path.join(data_path, self.CSV_FILENAME))
            self.data, self.nested_data = self.organize_data(os.path.join(data_path, self.DATA_FOLDERNAME))

        self.keylist = list(self.data.keys())

        self.object_cache = object_cache
        self.object_cache_path = os.path.join(self.models_path, "objects.npz")
        if self.object_cache not in [False, None]:
            if self.object_cache is True: # We may have gotten it from a split
                if not os.path.exists(self.object_cache_path):
                    self.make_object_cache()
                else:
                    self.object_cache = np.load(self.object_cache_path, allow_pickle=True)
        else:
            self.object_cache = None

    def organize_objects(self, objects_path):
        object_lists = np.genfromtxt(objects_path, delimiter=',', skip_header=1, dtype=None, encoding="utf-8")
        objects = []
        for object_list in object_lists:
            objects_dict = {header : value for header, value in zip(self.INFO_HEADERS, object_list)}
            objects.append(objects_dict)
        return objects

    def get_mesh(self, object_id):
        if self.object_cache is not None:
            return self.object_cache[f"{object_id}"].item()
        object_info = self.objects[object_id]
        location = object_info["location"].split("/")[-1]
        visual_dae_path = os.path.join(self.models_path, location, "visual_meshes", "visual.dae")
        mesh = trimesh.load(visual_dae_path, force="mesh")
        return mesh

    def get_info(self, object_id):
        return self.objects[object_id]
    
    def make_object_cache(self):
        object_infos = []
        object_meshes = {}
        for i in range(len(self.objects)):
            object_infos.append(self.get_info(i))
            object_meshes[f"{i}"] = self.get_mesh(i)
        np.savez(os.path.join(self.object_cache_path), **object_meshes, info=np.array(object_infos))
        self.object_cache = np.load(self.object_cache_path, allow_pickle=True)

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
                normalizer = None
                # normalizer = 255 if component == "color" \
                #         else 1000 if component == "depth" \
                #         else normalizer
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

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data, self.object_cache))

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

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data, self.object_cache))

    def create_organized_dataset_to_disk(self, dataset_folder, levels=None, split=None, mesh_samples=None):
        
        pose_data = self
        if split is not None:
            pose_data = self.txt_split(split)
        if levels is not None:
            pose_data = self.level_split(levels)

        os.makedirs(dataset_folder)

        shutil.copy(self.object_cache_path, os.path.join(dataset_folder, "objects.npz"))
        
        print("Scenes")
        scene_path = os.path.join(dataset_folder, "scenes")
        os.makedirs(scene_path)
        length = len(list(self.data.keys()))
        for i, key in enumerate(self.data.keys()):

            l, s, v = key

            scene = pose_data.data[key]

            color = scene["color"]() # normalization 255
            depth = scene["depth"]() # normalization 1000
            label = scene["label"]()
            meta = scene["meta"]
            projection = back_project(depth, meta)

            scene_path_i = os.path.join(scene_path, f"{l}-{s}-{v}")

            np.savez(scene_path, color=color, depth=depth, label=label, meta=meta, projection=projection)

            assert 0

            # if i % 100 == 0 :
            #     print(f"{i} / {length}")
            

            # object_ids = [object_id for object_id in np.unique(label) if object_id < 79]

            # for j, object_id in enumerate(object_ids):

            #     data_string = f"{i+j}_{l}-{s}-{v}_{object_id}"

            #     datum_path = os.path.join(dataset_folder, data_string)
            #     os.makedirs(datum_path)

            #     indices = np.where(label[key] == object_id)
            #     target_pcd = back_projection[indices]

            #     sample_count = len(target_pcd) if mesh_samples is None else mesh_samples
            #     source_pcd, faces = trimesh.sample.sample_surface(self.source_meshes[object_id], sample_count)
            #     source_pcd = source_pcd * self.metas[key]["scales"][object_id]
            #     # target_pcd  = target_pcd / self.metas[key]["scales"][object_id]

            #     rgb_crop = crop_image_using_segmentation(rgb, indices)

            #     try:
            #         target_pose = self.metas[key]["poses_world"][object_id][:3, :] # 4x4 -> 3x4
            #     except KeyError:
            #         target_pose = 0 # No Pose Provided; torch batching doens't allow None

            #     data_components = [

            #     ]

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, models_path, levels=None, split=None, mesh_samples=None):
        
        pose_data = PoseData(data_path, models_path)

        if split is not None:
            pose_data = pose_data.txt_split(split)
        if levels is not None:
            pose_data = pose_data.level_split(levels)

        self.pose_data = pose_data
        self.mesh_samples = mesh_samples

        # Create Caches
        NUM_CLASSES = len(pose_data.objects)
        self.object_cache = [False] * NUM_CLASSES
        self.object_infos = [None] * NUM_CLASSES
        self.source_meshes = [None] * NUM_CLASSES

        self.object_to_id = {}
        for i, object_info in enumerate(self.pose_data.objects):
            self.object_to_id[object_info["object"]] = i

        self.scene_cache = {}
        self.scene_count = len(pose_data.keys())
        self.rgbs = {}
        self.depths = {}
        self.labels = {}
        self.back_projections = {}

        self.length = 0
        self.metas = {}
        self.keys = []
        self.object_ids = []
        for i, key in enumerate(self.pose_data.keys()):
            self.scene_cache[key] = False
            self.metas[key] = self.pose_data[key]["meta"]
            objects = self.metas[key]["object_names"]
            count = len(objects)

            self.keys += [key] * count
            self.object_ids += [self.object_to_id[item] for item in objects]
            self.length += count

        self.point_cloud_cache = [False] * self.length
        self.indices = [None] * self.length
        self.source_points = [None] * self.length # From Models ; Baseline Objects
        self.target_points = [None] * self.length # Back Projected ; Off-Pose
        self.target_poses = [None] * self.length # Ground Truth Poses of Target Points


    def __len__(self):
        return len(self.object_ids)
    
    def __getitem__(self, i):

        object_id = self.object_ids[i]
        key = self.keys[i]
        
        self.cache_scene_data(key)
        self.cache_point_clouds(i, object_id, key)

        extras = (
            i, # index
            object_id,
            key, # scene key (level, scene, variant)
            self.object_infos[object_id], # info
            # self.source_meshes[object_id], # mesh
        )

        source_pcd = self.source_points[i]
        target_pcd = self.target_points[i]
        target_pose = self.target_poses[i]

        return source_pcd, target_pcd, target_pose, extras

    def cache_scene_data(self, key):
        if self.scene_cache[key] is not False:
            return
        self.scene_cache[key] = True
        
        scene = self.pose_data[key]
        self.rgbs[key] = scene["color"]()
        self.depths[key] = scene["depth"]()
        self.labels[key] = scene["label"]()
        self.back_projections[key] = back_project(self.depths[key], self.metas[key])

    def cache_point_clouds(self, i, object_id, key):
        if self.point_cloud_cache[i] is not False:
            return
        self.point_cloud_cache[i] = True
        
        indices = np.where(self.labels[key] == object_id)

        # False Register
        if len(indices[0]) == 0:
            print(f"MisRegister: {key} : {object_id}")

        self.indices[i] = indices

        target_pcd = self.back_projections[key][indices]

        self.cache_model_data(object_id) 

        sample_count = len(target_pcd) if self.mesh_samples is None else self.mesh_samples
        source_pcd, faces = trimesh.sample.sample_surface(self.source_meshes[object_id], sample_count)
        source_pcd = source_pcd * self.metas[key]["scales"][object_id]

        try:
            target_pose = self.metas[key]["poses_world"][object_id][:3, :] # 4x4 -> 3x4
        except KeyError:
            target_pose = 0 # No Pose Provided; torch batching doens't allow None
        
        self.source_points[i] = torch.tensor(source_pcd).float()
        self.target_points[i] = torch.tensor(target_pcd).float()
        self.target_poses[i] = target_pose if target_pose != 0 else 0


    def cache_model_data(self, object_id):
        if self.object_cache[object_id] is not False:
            return
        self.object_cache[object_id] = True
        
        self.object_infos[object_id] = self.pose_data.get_info(object_id)
        self.source_meshes[object_id] = self.pose_data.get_mesh(object_id)  
