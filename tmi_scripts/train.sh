
SEED=1
dataset=camus
device=0

############## DSNT-AL ################

 python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=dsnt-al
 python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=dsnt-al task.model.drop_block=True

# ############## DSNT-SKEW ################
# !!!!!!!!!! Not included in paper  !!!!!!!!!!
python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}]  task=dsnt-skew5
python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}]  task=dsnt-skew9

python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}]  task=dsnt-skew5 task.model.drop_block=True
python runner.py predict=False seed=${SEED} data=${dataset}-cont ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}]  task=dsnt-skew9 task.model.drop_block=True

############## SSN ################

python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=ssn
python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=ssn task.model.drop_block=True

################ Aleatoric ################

python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=aleatoric
python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=aleatoric task.model.drop_block=True

############### Segmentation ###################

python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=mcdropout task.model.drop_block=False
python runner.py predict=False seed=${SEED} data=${dataset} ++comet_tags=[${dataset},TRAIN] ++trainer.devices=[${device}] task=mcdropout task.model.drop_block=True



