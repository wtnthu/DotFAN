export CUDA_VISIBLE_DEVICES=0
python3 main.py \
--mode="train" \
--batch_size=5 \
--save_dir="v20" \
--L_pretrained="emore_recognition.pth" \
--P_pretrained="pose_precise_gray.pth.tar" \
--lambda_id_l2=8.0 \
--d_train_repeat=1 \
--gray="True" \
--num_epochs=30 \
--num_epochs_decay=15 \
--loss_rec_with_mask="True" \
--c_dim=240 \
--expression_w=1.0