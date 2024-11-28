# train on the synthetic dataset wine
CUDA_VISIBLE_DEVICES=0 python train.py --seed_set 425 --net_lr 1e-3  \
--use_3dgs --use_spike --use_flip  --use_multi_net --use_multi_reblur \
--data_name wine --exp_name joint_optimization --data_path data/synthetic/wine

# train on the real-world dataset sheep
CUDA_VISIBLE_DEVICES=0 python train.py --seed_set 425 --net_lr 1e-3  \
--use_3dgs --use_spike --use_flip  --use_multi_net --use_multi_reblur --use_real \
--data_name sheep --exp_name joint_optimization --data_path data/real_world/sheep

# render on the trained model
CUDA_VISIBLE_DEVICES=0 python render.py interpolate \
  --load-config outputs/sheep/bad-gaussians/<exp_date_time>/config.yml \
  --pose-source train \
  --frame-rate 30 \
  --output-format video \
  --interpolation-steps 5 \
  --output-path renders/sheep.mp4