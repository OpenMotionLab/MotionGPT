# parallel render
for i in `seq 0 7`
do
    for j in `seq 0 1`
    do
        CUDA_VISIBLE_DEVICES=$i python -m fit --dir $1 --save_folder $2 --cuda True &
        echo $j &
    done
done

wait 
echo "all weakup"

# # parallel render
# for i in `seq 0 25`
# do
#     CUDA_VISIBLE_DEVICES=$3 python -m fit --dir $1 --save_folder $2 --cuda True &
#     echo $i
# done
# wait 
# echo "all weakup"


# # gpu parallel render
# for i in `seq 0 7`
# do
#     CUDA_VISIBLE_DEVICES=$i python -m fit --dir $1 --save_folder $2 --cuda True &
#     echo $i
# done
# wait 
# echo "all weakup"
