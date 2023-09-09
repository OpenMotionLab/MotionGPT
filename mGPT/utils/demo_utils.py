import os
from pathlib import Path


# load example data
def load_example_input(txt_path):
    file = open(txt_path, "r")
    Lines = file.readlines()
    count = 0
    texts, lens = [], []
    # Strips the newline character
    for line in Lines:
        count += 1
        s = line.strip()
        s_l = s.split(" ")[0]
        s_t = s[(len(s_l) + 1):]
        lens.append(int(s_l))
        texts.append(s_t)
        print("Length-{}: {}".format(s_l, s_t))
    return texts, lens


# render batch
def render_batch(npy_dir, execute_python="./scripts/visualize_motion.sh", mode="sequence"):
    os.system(f"{execute_python} {npy_dir} {mode}")


# render
def render(execute_python, npy_path, jointtype, cfg_path):
    # execute_python = "/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender"
    # execute_python = "/apdcephfs/share_1227775/mingzhenzhu/jiangbiao/libs/blender-2.93.2-linux-x64/blender"
    export_scripts = "render.py"

    os.system(
        f"{execute_python} --background --python {export_scripts} -- --cfg={cfg_path} --npy={npy_path} --joint_type={jointtype}"
    )

    fig_path = Path(str(npy_path).replace(".npy", ".png"))
    return fig_path


# origin render
# def render(npy_path, jointtype):
#     execute_python = '/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender'
#     export_scripts = 'render.py'

#     os.system(f"{execute_python} --background --python {export_scripts} -- npy={npy_path} jointstype={jointtype}")

#     fig_path = Path(str(npy_path).replace(".npy",".png"))
#     return fig_path

# export fbx with hand params from pkl files
# refer to /apdcephfs/share_1227775/shingxchen/AIMotion/TMOST/scripts/fbx_output_smplx.py
def export_fbx_hand(pkl_path):
    input = pkl_path
    output = pkl_path.replace(".pkl", ".fbx")

    execute_python = "/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender"
    export_scripts = "./scripts/fbx_output_smplx.py"
    os.system(
        f"{execute_python} -noaudio --background --python {export_scripts}\
                --input {input} \
                --output {output}"
    )


# export fbx without hand params from pkl files
# refer to /apdcephfs/share_1227775/shingxchen/AIMotion/TMOST/scripts/fbx_output.py
def export_fbx(pkl_path):
    input = pkl_path
    output = pkl_path.replace(".pkl", ".fbx")

    execute_python = "/apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender"
    export_scripts = "./scripts/fbx_output.py"
    os.system(
        f"{execute_python} -noaudio --background --python {export_scripts}\
                --input {input} \
                --output {output}"
    )
