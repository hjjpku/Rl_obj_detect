require 'torch'

torch.class('tree_node')

function tree_node:__init()
	self.conv_fea = {}
	self.history_vector = {}
	self.detected_obj_table = {}
	self.cur_mask = {}
	self.iou_table = {}
	self.input_vector = {}
	self.new_iou = {}
	self.old_iou = {}
end