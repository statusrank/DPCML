CUDA_VISIBLE_DEVICES=0 python3 train_best.py \
    --data_path=data/Steam-200k \
    --model=COCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=5 \
    --sampling_strategy=uniform \
    --dim=100 \
    --reg=10  \
    --epoch=100 \
    --m1=0.1 \
    --m2=0.35


CUDA_VISIBLE_DEVICES=0 python3 train_best.py \
    --data_path=data/Steam-200k \
    --model=HarCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=5 \
    --sampling_strategy=hard \
    --dim=100 \
    --reg=10  \
    --epoch=100 \
    --m1=0.05 \
    --m2=0.25

CUDA_VISIBLE_DEVICES=0 python3 test_max_diversification.py \
    --data_path=data/Steam-200k \
    --model=COCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=5 \
    --sampling_strategy=uniform \
    --dim=100 \
    --reg=10  \
    --epoch=100 \
    --m1=0.1 \
    --m2=0.35


CUDA_VISIBLE_DEVICES=0 python3 test_max_diversification.py \
    --data_path=data/Steam-200k \
    --model=HarCML \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=5 \
    --sampling_strategy=hard \
    --dim=100 \
    --reg=10  \
    --epoch=100 \
    --m1=0.05 \
    --m2=0.25