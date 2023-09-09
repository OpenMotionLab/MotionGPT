# for npy folder
# CUDA_VISIBLE_DEVICES=0  /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=$1 --mode=$2

for j in `seq 0 2`
do
    CUDA_VISIBLE_DEVICES=0  /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --dir=$1 --mode=$2
done

# for single npy
# /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --cfg=./configs/render_cx.yaml --npy=$1 --joint_type=HumanML3D 
