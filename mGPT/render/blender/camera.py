import bpy


class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects['Camera']

        ## initial position
        camera.location.x = 7.36
        camera.location.y = -6.93
        if is_mesh:
            # camera.location.z = 5.45
            camera.location.z = 5.6
        else:
            camera.location.z = 5.2

        # wider point of view
        if mode == "sequence":
            if is_mesh:
                camera.data.lens = 65
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 85
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                # avoid cutting person
                camera.data.lens = 85
                # camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
