import tensorflow as tf
import numpy as np
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import TimeStep

action_count = 379
observation_count = 1402
fake_action_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.int32,
	name='action_mask',
	minimum=np.array(0),
	maximum=np.array(action_count - 1))
step_type_spec = TensorSpec(
	shape=(),
	dtype=tf.int32,
	name='step_type')
reward_type_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='reward',
	minimum=np.array(-1., dtype=np.float32),
	maximum=np.array(1., dtype=np.float32))
discount_type_spec = BoundedTensorSpec(
	shape=(),
	dtype=tf.float32,
	name='discount',
	minimum=np.array(0., dtype=np.float32),
	maximum=np.array(1., dtype=np.float32))
obs_type_spec = BoundedTensorSpec(
	shape=(observation_count,),
	dtype=tf.int32,
	name='observation',
	minimum=np.array(0),
	maximum=np.array(128))
mask_type_spec = BoundedTensorSpec(
	shape=(action_count,),
	dtype=tf.int32,
	name='action_mask',
	minimum=np.array(0),
	maximum=np.array(1))
fake_time_step_spec = TimeStep(
	step_type_spec,
	reward_type_spec,
	discount_type_spec,
	(obs_type_spec, mask_type_spec))
