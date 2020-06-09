python train.py --config DSCCass.json --run_id DSCCass
python test.py --resume saved/models/DSC/DSCCass/model_best.pth --dataset low
python test.py --resume saved/models/DSC/DSCCass/model_best.pth --dataset high
python test.py --resume saved/models/DSC/DSCCass/model_best.pth --dataset all

python train.py --config DSCAlt3.json
python test.py --resume saved/models/DSC/DSCAlt3/model_best.pth --dataset low
python test.py --resume saved/models/DSC/DSCAlt3/model_best.pth --dataset high
python test.py --resume saved/models/DSC/DSCAlt3/model_best.pth --dataset all

python train.py --config DSCAlt5.json
python test.py --resume saved/models/DSC/DSCAlt5/model_best.pth --dataset low
python test.py --resume saved/models/DSC/DSCAlt5/model_best.pth --dataset high
python test.py --resume saved/models/DSC/DSCAlt5/model_best.pth --dataset all