model_path=${HOME}/bushfire/u5155914/models/Paint-by-Example/model.ckpt
python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt ${model_path} \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 321 \
--scale 5
