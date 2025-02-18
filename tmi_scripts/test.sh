

SEED=1
dataset=camus
TAG=TMI_FINAL_TEST
device=2

############### DSNT-AL ################

python runner.py train=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=dsnt-al
python runner.py train=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=dsnt-al task.sequence_sampler=True

python runner.py train=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=dsnt-al task.model.drop_block=True task.t_e=10
python runner.py train=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=dsnt-al task.model.drop_block=True task.sequence_sampler=True task.t_e=10

############### SSN ################

 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=ssn
 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=ssn task.model.drop_block=True task.t_e=10

 ################ Aleatoric ################

 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=aleatoric
 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=aleatoric task.model.drop_block=True task.t_e=10

# ############### TTA ###################

 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=tta task.model.drop_block=False
 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=tta task.model.drop_block=True task.t_e=10

# ############### MC ###################

 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=mcdropout task.model.drop_block=True task.t_e=10
 python runner.py train=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},${TAG},${SEED}] ++trainer.devices=[${device}] task=mcdropout task.model.drop_block=True task.t_e=50
