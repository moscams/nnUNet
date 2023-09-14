cd /root/nnUNet/dataset/Spleen

python -m monai.apps.nnunet nnUNetV2Runner convert_dataset --input_config "./input.yaml"

python -m monai.apps.nnunet nnUNetV2Runner plan_and_process --input_config "./input.yaml"

python -m monai.apps.nnunet nnUNetV2Runner train_single_model --input_config "./input.yaml" --config "2d" --fold 0 --gpu_id 0
