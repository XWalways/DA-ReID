python main.py --mode train --backbone resnest50 --data_path Market-1501-v15.09.15 --labelsmooth --triplet --center \
	--kd_type avgloss --teacher teacher_market.pth --stage 1 --save_path weights --num_cls 751

#python main.py --mode train --backbone resnest50 --data_path Market-1501-v15.09.15 --labelsmooth --triplet --center \
#	--kd_type avgloss --teacher teacher_market.pth --stage 2 --save_path weights --num_cls 751

#python main.py --mode train --backbone resnest50 --data_path Market-1501-v15.09.15 --labelsmooth --triplet --center \
#	--kd_type avgloss --teacher teacher_market.pth --stage 3 --save_path weights --num_cls 751

#python main.py --mode evaluate --backbone resnest50 --data_path Market-1501-v15.09.15 --labelsmooth --triplet --center \
#	--kd_type avgloss --teacher teacher_market.pth --save_path weights/ --num_cls 751 --weight dareid_stage3_400.pt
