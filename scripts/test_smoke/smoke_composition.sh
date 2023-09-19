model_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/model.ckpt
seed=1
# 321
image_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/background_images/background.jpg
mask_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/background_images/mask_small.jpg

## declare an array variable
declare -a arr=("0adda456-f86b-4bd3-bacb-be3918a0aa89.jpeg"
                "24e5e3fd-e440-4a31-b315-ae1b6c701918.jpeg"
                "67304204-b65a-455b-a7df-37062121ab28.jpeg"
                )

## now loop through the above array
for img in "${arr[@]}"
do
    img_name=${img::-4}
    reference_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/reference_images/${img}
    output_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/output_results/figlib_big_smoke_small_${img_name}
    for ((seed=1; seed<=5; seed++))
    do
        python scripts/inference.py \
            --plms --outdir ${output_path}/${seed} \
            --config configs/v1.yaml \
            --ckpt ${model_path} \
            --image_path ${image_path} \
            --mask_path ${mask_path} \
            --reference_path ${reference_path} \
            --seed ${seed} \
            --scale 5
    done
done
