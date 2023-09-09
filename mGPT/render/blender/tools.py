import bpy
import numpy as np


def style_detect(data):
    is_mesh = False
    is_smplx = False
    jointstyle = 'mmm'
    # heuristic
    if data.shape[1] > 1000:
        is_mesh = True
    if data.shape[1] == 10475:
        is_smplx = True
    if data.shape[1] == 22:
        jointstyle =  'humanml3d'
        
    return is_mesh, is_smplx, jointstyle



# see this for more explanation
# https://gist.github.com/iyadahmed/7c7c0fae03c40bd87e75dc7059e35377
# This should be solved with new version of blender
class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0


def load_numpy_vertices_into_blender(vertices, faces, name, mat):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces.view(ndarray_pydata))
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    obj.active_material = mat
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action='DESELECT')
    return True


def delete_objs(names):
    if not isinstance(names, list):
        names = [names]
    # bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        for name in names:
            if obj.name.startswith(name) or obj.name.endswith(name):
                obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
