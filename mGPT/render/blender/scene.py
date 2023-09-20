import bpy
from .materials import plane_mat  # noqa


def setup_renderer(denoising=True, oldrender=True, accelerator="gpu", device=[0]):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.data.scenes[0].render.engine = "CYCLES"
    if accelerator.lower() == "gpu":
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        i = 0
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if i in device:  # gpu id
                d["use"] = 1
                print(d["name"], "".join(str(i) for i in device))
            else:
                d["use"] = 0
            i += 1

    if denoising:
        bpy.context.scene.cycles.use_denoising = True

    try:
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256
    except AttributeError as e:
        print(e)
        bpy.context.scene.cycles.tile_size = 256
    bpy.context.scene.cycles.samples = 64
    # bpy.context.scene.cycles.denoiser = 'OPTIX'

    if not oldrender:
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.display_settings.display_device = "sRGB"
        bpy.context.scene.view_settings.gamma = 1.2
        bpy.context.scene.view_settings.exposure = -0.75


# Setup scene
def setup_scene(
    res="high", denoising=True, oldrender=True, accelerator="gpu", device=[0]
):
    scene = bpy.data.scenes["Scene"]
    assert res in ["ultra", "high", "med", "low"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280 // 2
        scene.render.resolution_y = 1024 // 2
    elif res == "low":
        scene.render.resolution_x = 1280 // 4
        scene.render.resolution_y = 1024 // 4
    elif res == "ultra":
        scene.render.resolution_x = 1280 * 2
        scene.render.resolution_y = 1024 * 2

    scene.render.film_transparent= True
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(
        type="SUN", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.data.objects["Sun"].data.energy = 1.5

    # rotate camera
    bpy.ops.object.empty_add(
        type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.ops.transform.resize(
        value=(10, 10, 10),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )
    bpy.ops.object.select_all(action="DESELECT")

    setup_renderer(
        denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device
    )
    return scene
