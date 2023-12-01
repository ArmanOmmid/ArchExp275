import os
import shutil
import pickle
import numpy as np
from PIL import Image
import trimesh

import torch

from .utils import back_project, fps, crop_and_resize, crop_and_resize_multiple
from .pose_data import PoseData

class PoseDataNPZ():

    _npz_handlers_cache = {}

    def __init__(self, npz_data_path, data_path=None, models_path=None, levels=None, split=None, make_object_cache=False) -> None:
        
        self.npz_data_path = npz_data_path
        if data_path is not None and models_path is not None:
            self.pose_data = PoseData(data_path, models_path, make_object_cache=make_object_cache)
        else:
            assert os.path.exists(self.npz_data_path), "Must Provide NPZ Path if not providing data_path and model_path"
            print(f"Presumed Preloaded NPZ Dataset: {npz_data_path}")
            self.pose_data = None
            # NOTE : You cannot INTERNALLY do levels or splits this way

        self.npz(npz_data_path)
        
        self.objects_npz_path = os.path.join(npz_data_path, "objects.npz")
        if os.path.exists(self.objects_npz_path):
            # Cache Handlers
            if  self.objects_npz_path not in self._npz_handlers_cache:
                self._npz_handlers_cache[self.objects_npz_path] = np.load(os.path.join(npz_data_path, "objects.npz"), allow_pickle=True, mmap_mode="r")
            self.objects = self._npz_handlers_cache[self.objects_npz_path]
            self.info = self.objects["info"] # objects.csv
        else:
            self.info = self.pose_data.objects
            self.objects = None # Will have to get it manually from PoseData.get_mesh()

        self.object_RAM_cache = [None] * len(self.info)

        if levels is not None:
            levels = [levels] if isinstance(levels, int) else levels

        scenes_path = os.path.join(npz_data_path, "scenes")
        self.data = {}
        for file in os.listdir(scenes_path):
            key = tuple(int(i) for i in file.split(".")[0].split("-"))
            l, s, v = key
            if levels is not None and l not in levels:
                continue
            scene_path = os.path.join(npz_data_path, "scenes", f"{l}-{s}-{v}.npz")

            # Cache Handlers
            if scene_path not in self._npz_handlers_cache:
                self._npz_handlers_cache[scene_path] = np.load(scene_path, allow_pickle=True, mmap_mode="r") # NPZ Generator object 
            self.data[key] = self._npz_handlers_cache[scene_path]

            # color, depth, label, meta

        self.keylist = list(self.data.keys())

    def npz(self, npz_data_path):
        if self.pose_data is None:
            return
        self.npz_data_path = npz_data_path
        if os.path.exists(self.npz_data_path):
            print(f"NPZ Path Already Exists: {self.npz_data_path}")
            return
        self.pose_data.npz(self.npz_data_path)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.keylist)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.data[self.keylist[i]] # if you give an int
        else:
            return self.data[i] # if you give a key tuple (l, s, v)
        
    def get_mesh(self, obj_id):
        if self.object_RAM_cache[obj_id] is not None:
            return self.object_RAM_cache[obj_id]
        elif isinstance(self.objects, np.lib.npyio.NpzFile):
            mesh = self.objects[f"{obj_id}"].item()
        elif self.objects is None:
            mesh = self.pose_data.get_mesh(obj_id)
        
        self.object_RAM_cache[obj_id] = mesh # Cache the mesh!
        return mesh

    def get_info(self, obj_id):
        if self.info is None:
            return self.pose_data.get_info(obj_id)
        return self.info[obj_id]
    
    def sample_mesh(self, obj_id, n):
        return trimesh.sample.sample_surface(self.get_mesh(obj_id), n)[0] # samples, faces
    
    def meta(self, key):
        return self.data[key]["meta"][()]
    
    # def __del__(self):
    #     try:
    #         self.objects.close()
    #     except:
    #         pass
    #     for loader in self.data.values():
    #         loader.close()

class PoseDataNPZTorch(torch.utils.data.Dataset):
    def __init__(self, npz_data_path, data_path=None, models_path=None, 
                 levels=None, split=None, samples=8_000,
                 resize=(432, 768), aspect_ratio=True, margin=8): # Helps bigger images, worse on small images

        assert samples is not None, "No Longer Supporting Variable Samples"

        self.data = PoseDataNPZ(npz_data_path, data_path, models_path, levels, split)
        self.num_classes = len(self.data.info)
        self.samples = samples

        self.resize = resize
        self.aspect_ratio = aspect_ratio
        self.margin = margin

        self.source_pcd_cache = [None] * self.num_classes

        self._data = []

        for i, key in enumerate(self.data.keylist):
            for obj_id in self.data.meta(key)["objects"]:
                self._data.append((key, obj_id))

    def __len__(self):
        return len(self.data)
    
    def sample_source_pcd(self, obj_id):
        if self.source_pcd_cache[obj_id] is None:
            self.source_pcd_cache[obj_id] = \
                self.data.sample_mesh(obj_id, self.samples).astype(np.float32)
            
        return self.source_pcd_cache[obj_id]
        
    def __getitem__(self, i):
        key, obj_id = self._data[i]

        scene = self.data[key]

        color = scene["color"]
        depth = scene["depth"]
        # label = scene["label"]
        mask = scene["label"] == obj_id
        meta = scene["meta"][()]


        (color, depth, mask), scale, translate = crop_and_resize_multiple(
            (color, depth, mask), 
            mask, target_size=self.resize, margin=self.margin, aspect_ratio=self.aspect_ratio)
        
        color = color.astype(np.float32) / 255
        depth = depth.astype(np.float32) / 1000

        color = np.transpose(color, (2, 0, 1)) # H W C -> C H W # NOTE : Do this after crop/resize

        target_pcd = back_project(depth, meta, mask, (scale, translate)).astype(np.float32)

        t_samples = len(target_pcd)
        if t_samples > self.samples:
            point_indices = np.linspace(start=0, stop=len(target_pcd)-1, num=self.samples, dtype=int)
            target_pcd = target_pcd[point_indices] # Sa
        elif t_samples <= self.samples:
            # point_indices = np.arange(t_samples) # Get true point indices, ignoring repeats
            if t_samples < self.samples:
                point_indices = np.arange(t_samples)
                repeats = np.ceil(self.samples / t_samples).astype(int)
                point_indices = np.repeat(point_indices, repeats, axis=0)[:self.samples]
                target_pcd = target_pcd[point_indices]
            else:
                point_indices = np.arange(self.samples) # for batching, we need the repeats.
            # If we do repeats, when we concatonate with RGB, RGB is limited to the original set
            # So we just take the original points and concat and get rid of all duplicates

        point_count = t_samples # this helps us figure out batching

        source_pcd = self.sample_source_pcd(obj_id) * meta["scales"][obj_id]
        pose = meta["poses_world"][obj_id][:3, :] # 4x4 -> 3x4

        mask_info = (mask, point_count, point_indices)

        return source_pcd, target_pcd, color, depth, mask_info, pose
