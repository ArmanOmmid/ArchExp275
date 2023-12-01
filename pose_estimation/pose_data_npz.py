import os
import shutil
import pickle
import numpy as np
from PIL import Image
import trimesh

import torch

from .utils import back_project, crop_image_using_segmentation, fps
from .pose_data import PoseData

class PoseDataNPZ():
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
            self.objects = np.load(os.path.join(npz_data_path, "objects.npz"), allow_pickle=True)
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
            self.data[key] = np.load(scene_path, allow_pickle=True) # NPZ Generator object 
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

class PoseDataNPZTorch(torch.utils.data.Dataset):
    def __init__(self, npz_data_path, data_path=None, models_path=None, 
                 levels=None, split=None, mesh_samples=20_000):
        

        self.data = PoseDataNPZ(npz_data_path, data_path, models_path, levels, split)
        self.num_classes = len(self.data.info)
        self.mesh_samples = mesh_samples

        self.source_pcd_cache = [None] * self.num_classes

        self._data = []

        for i, key in enumerate(self.data.keylist):
            for obj_id in self.data.meta(key)["objects"]:
                self._data.append((key, obj_id))

    def __len__(self):
        return len(self.data)
    
    def sample_source_pcd(self, obj_id, n):
        if self.source_pcd_cache[obj_id] is None:
            n = n if self.mesh_samples is None else self.mesh_samples
            self.source_pcd_cache[obj_id] = \
                self.data.sample_mesh(obj_id, self.mesh_samples).astype(np.float32)
            
        return self.source_pcd_cache[obj_id]
        

    def __getitem__(self, i):
        key, obj_id = self._data[i]

        scene = self.data[key]

        # color = scene["color"] * 255
        # depth = scene["depth"] / 1000
        # label = scene["label"]
        meta = scene["meta"][()]
        projection = back_project(scene["depth"] / 1000, meta)

        target_pcd = projection[np.where(scene["label"] == obj_id)].astype(np.float32) # average = 3000
        source_pcd = self.sample_source_pcd(obj_id, len(target_pcd)) * meta["scales"][obj_id]
        pose = meta["poses_world"][obj_id]

        return source_pcd, target_pcd, pose











        
