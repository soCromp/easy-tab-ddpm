# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-adult1.toml --train --sample
# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-adult2.toml --train --sample
# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-adult3.toml --train --sample

# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-diabetes-new1.toml --train --sample
# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-diabetes-new2.toml --train --sample
# CUDA_VISIBLE_DEVICES=0 python ../scripts/pipeline.py --config cb-diabetes-new3.toml --train --sample

CUDA_VISIBLE_DEVICES=1 python ../scripts/pipeline.py --config cb-house1.toml --train --sample
CUDA_VISIBLE_DEVICES=1 python ../scripts/pipeline.py --config cb-house2.toml --train --sample
CUDA_VISIBLE_DEVICES=1 python ../scripts/pipeline.py --config cb-house3.toml --train --sample
