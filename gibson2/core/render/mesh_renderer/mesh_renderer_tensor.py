import cv2
import sys
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, GL
import torch
from gibson2.core.render.mesh_renderer.get_available_devices import get_cuda_device

class MeshRendererG2G(MeshRenderer):
    """
    Similar to MeshRenderer, but allows rendering to pytorch tensor, note that
    pytorch installation is required.
    """

    def __init__(self, width=512, height=512, fov=90, device_idx=0, use_fisheye=False):
        super(MeshRendererG2G, self).__init__(width, height, fov, device_idx, use_fisheye)
        self.cuda_idx = get_cuda_device(self.device_minor)
        print("Using cuda device {}".format(self.cuda_idx))
        with torch.cuda.device(self.cuda_idx):
            self.image_tensor = torch.cuda.ByteTensor(height, width, 4).cuda()
            self.normal_tensor = torch.cuda.ByteTensor(height, width, 4).cuda()
            self.seg_tensor = torch.cuda.ByteTensor(height, width, 4).cuda()
            self.pc_tensor = torch.cuda.FloatTensor(height, width, 4).cuda()

    def readbuffer_to_tensor(self, modes=('rgb', 'normal', 'seg', '3d')):
        results = []
        with torch.cuda.device(self.cuda_idx):
            if 'rgb' in modes:
                self.r.map_tensor(int(self.color_tex_rgb), int(self.width), int(self.height),
                                  self.image_tensor.data_ptr())
                results.append(self.image_tensor.clone())
            if 'normal' in modes:
                self.r.map_tensor(int(self.color_tex_normal), int(self.width), int(self.height),
                                  self.normal_tensor.data_ptr())
                results.append(self.normal_tensor.clone())
            if 'seg' in modes:
                self.r.map_tensor(int(self.color_tex_semantics), int(self.width), int(self.height),
                                  self.seg_tensor.data_ptr())
                results.append(self.seg_tensor.clone())
            if '3d' in modes:
                self.r.map_tensor_float(int(self.color_tex_3d), int(self.width), int(self.height),
                                  self.pc_tensor.data_ptr())
                results.append(self.pc_tensor.clone())

        return results
    def render(self, modes=('rgb', 'normal', 'seg', '3d'), hidden=()):
        """
        A function to render all the instances in the renderer and read the output from framebuffer into pytorch tensor.

        :param modes: it should be a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d').
        :param hidden: Hidden instances to skip. When rendering from a robot's perspective, it's own body can be
            hidden

        """
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        for instance in self.instances:
            if not instance in hidden:
                instance.render()
        GL.glDisable(GL.GL_DEPTH_TEST)

        return self.readbuffer_to_tensor(modes)