model_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/model.ckpt
seed=1
# 321
image_path=examples/figlib/background.jpg
mask_path=examples/figlib/mask.jpg
reference_path=examples/big_smoke/0adda456-f86b-4bd3-bacb-be3918a0aa89.jpeg
output_path=examples/figlib/output

for ((seed=1; seed<=5; seed++))
do
    python scripts/inference.py \
        --plms --outdir ${output_path} \
        --config configs/v1.yaml \
        --ckpt ${model_path} \
        --image_path ${image_path} \
        --mask_path ${mask_path} \
        --reference_path ${reference_path} \
        --seed ${seed} \
        --scale 5
done
