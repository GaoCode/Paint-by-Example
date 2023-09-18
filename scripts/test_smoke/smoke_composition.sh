model_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/model.ckpt
seed=321
python scripts/inference.py \
--plms --outdir smoke_test_results/figlib_big_smoke \
--config configs/v1.yaml \
--ckpt ${model_path} \
--image_path examples/figlib/background.jpg \
--mask_path examples/figlib/mask.jpg \
--reference_path examples/big_smoke/0adda456-f86b-4bd3-bacb-be3918a0aa89.jpeg \
--seed ${seed} \
--scale 5
