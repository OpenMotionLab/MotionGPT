# # parallel fit
# for i in `seq 0 7`
# do
#     for j in `seq 0 2`
#     do
#         CUDA_VISIBLE_DEVICES=$i python -m fit --dir $1 --save_folder $2 --cuda True &
#         echo $j &
#     done
# done
# wait 
# echo "all weakup"


# parallel render
for i in `seq 0 7`
do
    for j in `seq 0 2`
    do
        sleep 1 &
        CUDA_VISIBLE_DEVICES=$i /apdcephfs/share_1227775/shingxchen/libs/blender_bpy/blender-2.93.2-linux-x64/blender --background --python render.py -- --dir=$1 --mode=$2 &
        echo $i
    done
done
wait 
echo "all weakup"
