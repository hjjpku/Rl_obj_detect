require 'Hjj_image_loader'
require 'Hjj_read_cmd'
require 'Hjj_reinforcement'
require 'Hjj_feature_extractor'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'Hjj_tree_node'
require 'Hjj_check_iou'	
require 'Hjj_actions'

local cmd = torch.CmdLine()
opt = func_read_test_cmd(cmd, arg)

--********************************************************
-- ******************** DATA PREPARING ******************
-- saved test dataset
local testset_path = './test_data'
local path_voc1 = "./VOC2007_test/VOCdevkit/VOC2007"
local path_voc2 = "./VOC2012_test/VOCdevkit/VOC2012"
local path_voc
local result_path = './result'

if opt.dataset == 1 then
	path_voc = path_voc1
	testset_path = testset_path .. '/2007/'
	result_path = result_path .. '/2007/'
else
	path_voc = path_voc2
	testset_path = testset_path .. '/2012/'
	result_path = result_path .. '/2012/'
end

local images = {}
local annotations = {}
local names = {}

--###########################################################

--********************************************************
-- ******************** MODEL PREPARING ******************
local resnet_model = torch.load(opt.resnet)
resnet_model = resnet_model.resnet_model
--resnet_model = resnet_model.model
resnet_model:evaluate()
resnet_model = resnet_model:cuda()
local orig = resnet_model:get(#resnet_model.modules)
assert(torch.type(orig) == 'cudnn.SoftMax',
         'expected last layer to be SoftMax Layer')
local fc_weight = resnet_model:get(#resnet_model-1).weight:cuda()
local softmax_net = nn.SoftMax():cuda() -- for CAM

local dqn = torch.load(opt.dqn)
dqn = dqn.dqn
dqn:evaluate()
dqn = dqn:cuda()

local pvn = torch.load(opt.pvn)
pvn = pvn.pvn
pvn:evaluate()
pvn = pvn:cuda()

local rgn
if opt.enable_rgn == 1 then
	rgn = torch.load(opt.rgn)
	rgn = rgn.rgn
	rgn:evaluate()
	rgn = rgn:cuda()
end
--###########################################################

--********************************************************
-- ******************** DETECT SETTING ******************
local max_steps = opt.max_steps
local action1_alpha = 0.55
local action2_alpha = 0.25
local topk = opt.topk
local thd = 0.5
--###########################################################

--********************************************************
-- ************************* DETECT **********************
local total_target = 0
local total_found = 0
local total_file = io.open(result_path .. opt.name .. '.txt', 'w')
for cls = 1,1 do
	local target_num = 0
	local found_num = 0
	if opt.load_saved_testdata == 1 then
		local test_data = torch.load(testset_path .. cls .. '_images.t7')
		images = test_data.images
		test_data = torch.load(testset_path .. cls.. '_annotations.t7')
		annotations = test_data.annotations
		test_data = torch.load(testset_path .. cls.. '_names.t7')
		names = test_data.names

	else
		images, annotations, names = func_image_loader(0, path_voc, nil, cls)
		names = names[1] -- images is a {{...}} in test phase
		torch.save(testset_path .. cls .. '_images.t7', {images = images})
		torch.save(testset_path .. cls .. '_annotations.t7', {annotations = annotations})
		torch.save(testset_path .. cls .. '_names.t7', {names = names})
	end
	print('Loading test data ' .. cls ..' finished..')

	local prop_file = io.open(result_path .. cls .. opt.name .. '.txt', 'w')
	--local prop_nms_file = io.open(result_path .. cls .. opt.name .. '_nms.txt', 'w')
	local loop_time = 0
	for i,v in pairs(images) do
		local st = os.clock()
		local img_record_buff = {}
		local cur_annotation = {}
		for tt, detail in pairs(annotations[i]) do
			if detail[1] == cls then  
				table.insert(cur_annotation,detail)
			end
		end
		local gt_num = #cur_annotation
		if gt_num == 0 then
			print('BOOM!')
			exit()
		end
		local img_detected_record = torch.Tensor(gt_num):fill(0):int()
		local cur_img = v
		cur_img = func_image_processing_for_resnet(cur_img)
		local cur_img_size = {cur_img:size(2), cur_img:size(3)}

		local node_queue = {}
		local tmp_node = tree_node()

		tmp_node.cur_mask = {1,1, cur_img:size(2), cur_img:size(3)}
		tmp_node.detected_obj_table = torch.Tensor(gt_num):fill(0):int()
		local detected_obj_table = tmp_node.detected_obj_table:clone()

		tmp_node.iou_table = torch.Tensor(gt_num):fill(0)
		tmp_node.old_iou = 0
		tmp_node.new_iou = 0
		local index
		tmp_node.old_iou, tmp_node.new_iou, tmp_node.iou_table, index, tmp_node.detected_obj_table = 
						func_follow_iou(tmp_node.cur_mask,cur_annotation,tmp_node.detected_obj_table,
										tmp_node.iou_table, thd)
		local detected_obj_table = tmp_node.detected_obj_table:clone()
		img_detected_record:cbitor(tmp_node.detected_obj_table)

		local conv_fea, class_softmax = func_get_image_conv_feature_and_softmax(
																		resnet_model, cur_img, 
																		tmp_node.cur_mask)

		if opt.enable_CAM == 1 then
			conv_fea = func_conv_cam_fusion(conv_fea, fc_weight, class_softmax, nil, topk,softmax_net)
		end
		conv_fea = conv_fea:view(conv_fea:nElement())
		local history_vector = torch.Tensor(history_vector_size):fill(0):cuda()
		tmp_node.history_vector = torch.Tensor(history_vector_size):fill(0):cuda()
		tmp_node.conv_fea = conv_fea
		tmp_node.input_vector = torch.cat(tmp_node.conv_fea, tmp_node.history_vector, 1)
		table.insert(node_queue, tmp_node)

		local obj_y_o_n = pvn:forward(tmp_node.conv_fea)
		local tmp_v, pred_cls = torch.max(class_softmax,1)

		local tmp_record = {names[i],class_softmax[cls],tmp_node.cur_mask,obj_y_o_n[1],
							tmp_node.new_iou,pred_cls[1],tmp_v[1], img_detected_record:sum(), gt_num, index}
		table.insert(img_record_buff , tmp_record)
		func_write_prop_file(prop_file, tmp_record)
		func_write_prop_file(total_file, tmp_record)
		local step_count = 1

		while(step_count <= max_steps) do
			print('\t\tStep ' .. step_count .. ':\n')

			local cur_node = node_queue[1]
			print('\t\tCur_Mask = [ ' .. cur_node.cur_mask[1] .. ', ' .. 
				cur_node.cur_mask[2] .. ', ' .. cur_node.cur_mask[3] .. ', ' ..
				cur_node.cur_mask[4] .. ' ]\n')
			local action_output = dqn:forward(cur_node.input_vector)

			local tmp_v, action1 = torch.max(action_output[{{1,5}}],1)
			action1 = action1[1]-- from tensor to numeric type
				-- translation action
			local tmp_v, action2 = torch.max(action_output[{{6,number_of_actions}}],1)
			action2 = action2[1] + 5 -- from tensor to numeric type
			print(action_output)
			local new_node1 = tree_node()
			local now_target_gt1
			local class_softmax1
			local flag1
			local index1
			local new_node2 = tree_node()
			local now_target_gt2
			local class_softmax2
			local flag2
			local index2

			new_node1, reward1, now_target_gt1,flag1, history_vector, index1 = func_run_dqn_action(
													new_node1, cur_node, action1, 
													action1_alpha, cur_annotation, 
													thd,cur_img_size,history_vector,detected_obj_table)
			detected_obj_table = new_node1.detected_obj_table:clone()
			
			new_node2, reward2, now_target_gt2,flag2, history_vector, index2 = func_run_dqn_action(
													new_node2, cur_node, action2,
													action2_alpha, cur_annotation, 
													thd, cur_img_size, history_vector, detected_obj_table)	
			detected_obj_table = new_node2.detected_obj_table:clone()		

			new_node1.conv_fea, class_softmax1, new_node2.conv_fea, class_softmax2 =
											func_get_2_conv_feature_and_softmax(
														resnet_model, cur_img, 
														new_node1.cur_mask,
														new_node2.cur_mask)
			  
			if opt.enable_CAM == 1 then
				new_node1.conv_fea = func_conv_cam_fusion(
										new_node1.conv_fea, fc_weight, 
										class_softmax1, nil, topk,softmax_net)
				new_node2.conv_fea = func_conv_cam_fusion(
										new_node2.conv_fea, fc_weight, class_softmax2, 
										nil, topk,softmax_net)
			end

			if opt.enable_rgn == 1 then
				local loc = rgn:forward(new_node1.conv_fea)
				new_node1.cur_mask = func_localization_regression(loc, new_node1.cur_mask,cur_img_size)
				loc = rgn:forward(new_node2.conv_fea)
				new_node2.cur_mask = func_localization_regression(loc, new_node2.cur_mask,cur_img_size)
				new_node1.conv_fea, class_softmax1, new_node2.conv_fea, class_softmax2 =
											func_get_2_conv_feature_and_softmax(
														resnet_model, cur_img, 
														new_node1.cur_mask,
														new_node2.cur_mask)

				if opt.enable_CAM == 1 then
					new_node1.conv_fea = func_conv_cam_fusion(
											new_node1.conv_fea, fc_weight, 
											class_softmax1, nil, topk,softmax_net)
					new_node2.conv_fea = func_conv_cam_fusion(
											new_node2.conv_fea, fc_weight, class_softmax2, 
											nil, topk,softmax_net)
				end
			end

			new_node1.conv_fea = new_node1.conv_fea:view(new_node1.conv_fea:nElement())
			new_node2.conv_fea = new_node2.conv_fea:view(new_node2.conv_fea:nElement())
			new_node1.input_vector = torch.cat(new_node1.conv_fea, new_node1.history_vector, 1)
			new_node2.input_vector = torch.cat(new_node2.conv_fea, new_node2.history_vector, 1)

			img_detected_record:cbitor(new_node1.detected_obj_table)
			obj_y_o_n = pvn:forward(new_node1.conv_fea)
			tmp_v, pred_cls = torch.max(class_softmax1,1)
			tmp_record = {names[i],class_softmax1[cls],new_node1.cur_mask,obj_y_o_n[1],
							new_node1.new_iou,pred_cls[1],tmp_v[1],img_detected_record:sum(), gt_num, index1}
			table.insert(img_record_buff , tmp_record)
			func_write_prop_file(prop_file, tmp_record)
			func_write_prop_file(total_file, tmp_record)

			img_detected_record:cbitor(new_node2.detected_obj_table)
			obj_y_o_n = pvn:forward(new_node2.conv_fea)
			tmp_v, pred_cls = torch.max(class_softmax2,1)
			tmp_record = {names[i],class_softmax2[cls],new_node2.cur_mask,obj_y_o_n[1],
							new_node2.new_iou,pred_cls[1],tmp_v[1],img_detected_record:sum(), gt_num, index2}
			table.insert(img_record_buff , tmp_record)
			func_write_prop_file(prop_file, tmp_record)
			func_write_prop_file(total_file, tmp_record)

			table.insert(node_queue, new_node1)
			table.insert(node_queue, new_node2)
			table.remove(node_queue, 1)

			step_count = step_count + 2
			--if img_detected_record:sum() == gt_num then
			--	print('\t\t## All Objects founded in ' .. step_count .. 'steps\n')
			--end

		end -- while(step) loop 
		target_num = target_num + gt_num
		found_num = found_num + img_detected_record:sum()
		--func_nms(prop_nms_file, img_record_buff, 0.7)
		local et = os.clock()
		print('img time = ' .. et-st)
		loop_time = loop_time + et -st 	
	end
	io.close(prop_file)
	print('\tClass ' .. cls .. ' : find/total = ' .. found_num .. '/' .. target_num ..
		' ; Recall = ' .. found_num/target_num .. '\n' )

	total_found = total_found + found_num
	total_target = total_target + target_num
	print('loop_avg = ' .. loop_time/#images)
end
print('In All: find/total = ' .. total_found/total_target .. '\n')
io.close(total_file)
