import caffe
from fast_rcnn.config import cfg
from utils.timer import Timer
import numpy as np
from caffe.proto import caffe_pb2
import google.protobuf as pb2


class SolverWrapper(object):
	"""A simple wrapper around Caffe's solver.
	This wrapper gives us control over he snapshotting process, which we
	use to unnormalize the learned bounding-box regression weights.
	only normalize in proposal_target_layer, not in anchor_target_layer
	"""

	def __init__(self, solver_prototxt, roidb,
				 solverstate, pretrained_model):

		if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
			assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
			bbox_means = np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS, dtype=np.float32)
			bbox_stds = np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS, dtype=np.float32)
			num_classes = 2
			self.bbox_means = np.tile(bbox_means, (num_classes, 1)).ravel()
			self.bbox_stds = np.tile(bbox_stds, (num_classes, 1)).ravel()

		self.solver = caffe.SGDSolver(solver_prototxt)
		if solverstate is not None:
			self.solver.restore(solverstate)
		if pretrained_model is not None:
			self.solver.net.copy_from(pretrained_model)

		self.solver_param = caffe_pb2.SolverParameter()
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)

		self.solver.net.layers[0].set_roidb(roidb)

	def snapshot_caffemodel(self):
		"""Take a snapshot of the network after unnormalizing the learned
		bounding-box regression weights. This enables easy use at test-time.
		"""
		net = self.solver.net

		ori_param_map = {}
		candidate_blob_names=[x+"_rfcn_bbox" for x in cfg.CLASSES[1:]]
		candidate_blob_names.insert(0, "rfcn_bbox")

		for blob_name in candidate_blob_names:
			scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
									  net.params.has_key(blob_name))
			if scale_bbox_params_rfcn:
				# save original values
				orig_0 = net.params[blob_name][0].data.copy()
				orig_1 = net.params[blob_name][1].data.copy()
				# position_num**2
				repeat = orig_1.shape[0] / self.bbox_means.shape[0]

				# scale and shift with bbox reg unnormalization; then save snapshot, only impact proposal_target_layer, not rpn
				# rfcn_bbox: [position_num**2 * num_classes*4]   num_classes=2 if AGONISTIC
				net.params[blob_name][0].data[...] = \
					(net.params[blob_name][0].data *
					 np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
				net.params[blob_name][1].data[...] = \
					(net.params[blob_name][1].data *
					 np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))
				ori_param_map[blob_name] = {}
				ori_param_map[blob_name][0] = orig_0
				ori_param_map[blob_name][1] = orig_1

		filename = (
			self.solver_param.snapshot_prefix + 'unnormalized_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
		net.save(str(filename))


		for blob_name in ori_param_map.keys():
			scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
									  net.params.has_key(blob_name))
			if scale_bbox_params_rfcn:
				# restore net to original state
					net.params[blob_name][0].data[...] = ori_param_map[blob_name][0]
					net.params[blob_name][1].data[...] = ori_param_map[blob_name][1]

		return filename

	def train_model(self, max_iters):
		"""Network training loop."""
		last_snapshot_iter = -1
		timer = Timer()
		model_paths = []
		while self.solver.iter < max_iters:
			# Make one SGD update
			timer.tic()
			self.solver.step(1)
			timer.toc()
			if self.solver.iter % (10 * self.solver_param.display) == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)

			if self.solver.iter % cfg.TRAIN.UNNORMALIZED_SNAPSHOT_ITERS == 0:
				last_snapshot_iter = self.solver.iter
				model_paths.append(self.snapshot_caffemodel())

		if last_snapshot_iter != self.solver.iter:
			model_paths.append(self.snapshot_caffemodel())
		return model_paths


def train_net(solver_prototxt, roidb, solverstate, pretrained_model, max_iters=40000):
	sw = SolverWrapper(solver_prototxt, roidb, solverstate=solverstate, pretrained_model=pretrained_model)

	print 'Solving...'
	model_paths = sw.train_model(max_iters)
	print 'done solving'
	return model_paths
