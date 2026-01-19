conda activate AI

python simple_trainer_ours.py --data_dir ../data/LOM_full/bike --exp_name low --result_dir ../results/low/bike/
python simple_trainer_ours.py --data_dir ../data/LOM_full/bike --exp_name over_exp --result_dir ../results/high/bike/
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/low/buu/
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --result_dir ../results/high/buu/
python simple_trainer_ours.py --data_dir ../data/LOM_full/chair --exp_name low --result_dir ../results/low/chair/
python simple_trainer_ours.py --data_dir ../data/LOM_full/chair --exp_name over_exp --result_dir ../results/high/chair/
python simple_trainer_ours.py --data_dir ../data/LOM_full/shrub --exp_name low --result_dir ../results/low/shrub/
python simple_trainer_ours.py --data_dir ../data/LOM_full/shrub --exp_name over_exp --result_dir ../results/high/shrub/
python simple_trainer_ours.py --data_dir ../data/LOM_full/sofa --exp_name low --result_dir ../results/low/sofa/
python simple_trainer_ours.py --data_dir ../data/LOM_full/sofa --exp_name over_exp --result_dir ../results/high/sofa/

python simple_trainer_ours.py --data_dir ../data/NeRF_360/bicycle --exp_name variance --data_factor 8 --result_dir ../results/variance/bicycle/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/bonsai --exp_name variance --data_factor 8 --result_dir ../results/variance/bonsai/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/counter --exp_name variance --data_factor 8 --result_dir ../results/variance/counter/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/garden --exp_name variance --data_factor 8 --result_dir ../results/variance/garden/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/kitchen --exp_name variance --data_factor 8 --result_dir ../results/variance/kitchen/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/room --exp_name variance --data_factor 8 --result_dir ../results/variance/room/
python simple_trainer_ours.py --data_dir ../data/NeRF_360/stump --exp_name variance --data_factor 8 --result_dir ../results/variance/stump/