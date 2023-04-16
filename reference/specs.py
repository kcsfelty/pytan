import tensorflow as tf
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import TimeStep

action_count = 379
observation_count = 1402
action_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.int32,
	name='action_mask',
	minimum=0,
	maximum=action_count - 1)
step_spec = TensorSpec(
	shape=(),
	dtype=tf.int32,
	name='step_type')
reward_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='reward',
	minimum=-1,
	maximum=1)
discount_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='discount',
	minimum=0.,
	maximum=1.)
observation_spec = BoundedTensorSpec(
	shape=(observation_count,),
	dtype=tf.int32,
	name='observation',
	minimum=0,
	maximum=127)
mask_spec = BoundedTensorSpec(
	shape=(action_count,),
	dtype=tf.int32,
	name='action_mask',
	minimum=0,
	maximum=1)
time_step_spec = TimeStep(
	step_spec,
	reward_spec,
	discount_spec,
	(observation_spec, mask_spec))
