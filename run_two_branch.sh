if [ "$1" = "--train" ]; then
    CUDA_VISIBLE_DEVICES=0 \
    python3.5 train_embedding_nn.py \
    --image_feat_path /home/litongxin/image_attribute_two_branch/img_feat.mat \
    --sent_feat_path /home/litongxin/Two_branch_network/two_branch_img_feature.mat \
    --save_dir /home/litongxin/image_attribute_two_branch
fi

if [ "$1" = "--test" ]; then
    CUDA_VISIBLE_DEVICES=5 \
    python3.5 eval_embedding_nn.py \
    --image_feat_path /home/litongxin/image_attribute_two_branch/img_feat_test.mat \
    --sent_feat_path /home/litongxin/Two_branch_network/two_branch_img_feature.mat \
    --restore_path /home/litongxin/image_attribute_two_branch-876.meta
fi

if [ "$1" = "--val" ]; then
    CUDA_VISIBLE_DEVICES=1 \
    python eval_embedding_nn.py \
    --image_feat_path /path/to/image_feature_val.mat \
    --sent_feat_path /path/to/text_feature_val.mat \
    --restore_path /path/to/save_dir/
fi
