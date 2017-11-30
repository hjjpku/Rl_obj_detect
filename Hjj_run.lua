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
require 'optim'

local cmd = torch.CmdLine()
opt = func_read_training_cmd(cmd, arg)


--********************************************************
-- ******************** DATA PREPARING ******************

-- path of PASCAL VOC 2012 or other database to use for training
local path_voc = "./VOC2012_train/VOCdevkit/VOC2012"
-- path of other PASCAL VOC dataset, if you want to train with 2007 and 2012 train datasets
local path_voc2 = "./VOC2007_train/VOCdevkit/VOC2007"
-- path of voc test data
local path_test = './VOC2007_test/VOCdevkit/VOC2007'
-- saved training dataset
local dataset_path = "/home/zangxh/hjj_OD/rl_obj_detec/training_data/"
-- saved test dataset
local testset_path = './test_data'

local images ={}
local annotations = {}

if opt.load_saved_data == 0 then
	images, annotations = func_image_loader(opt.dataset, path_voc, path_voc2)
	torch.save(dataset_path .. 'images.t7', {images = images})
	torch.save(dataset_path .. 'annotations.t7', {annotations = annotations})
	print('Loading Training data finished..')
end



local test_images = {}
local test_annotations = {}
if opt.enable_evaluate == 1 then
	if opt.load_saved_testdata == 1 then
		local test_data = torch.load(testset_path .. 'images.t7')
		test_images = test_data.images
		test_data = torch.load(testset_path .. 'annotations.t7')
		test_annotations = test_data.annotations
	else
		test_images, test_annotations = func_image_loader(0, path_test, path_voc2)
		torch.save(testset_path .. 'images.t7', {images = test_images})
		torch.save(testset_path .. 'annotations.t7', {annotations = test_annotations})
	end
	print('Loading test data finished..')
end

--###########################################################


--********************************************************
-- ******************** RESNET-101 PREPARING ******************
local resnet_path = "./fb.resnet.torch-master/pretrained/model_17.t7"
if enable_background == 1 then
	resnet_path = "./fb.resnet.torch-master/21_class_resnet/model_12.t7"
end
local resnet_save_path = './fituned_resnet'
local resnet_model = torch.load(resnet_path)
resnet_model = resnet_model:cuda()
local fc_weight = resnet_model:get(#resnet_model).weight:cuda()
local softMaxLayer = cudnn.SoftMax():cuda()
-- add Softmax layer
resnet_model:add(softMaxLayer) 

local resnet_logger = optim.Logger(opt.resnet_log)

-- finetune setting for resnet
local resnet_params, resnet_gradParams = resnet_model:getParameters()
local resnet_criterion = nn.CrossEntropyCriterion():cuda()
local resnet_optimState = {learningRate = 0.0001, maxIteration = 1, learningRateDecay = 0.0001, evalCounter = 0}
local resnet_batch_size = 16
local resnet_buffer = {}

if opt.finetune_resnet == 0 then
	resnet_model:evaluate()
end
print('Loading Resnet-101 finished.. ')

local softmax_net = nn.SoftMax():cuda() -- for CAM
--###########################################################

--********************************************************
-- ******************** DQN PREPARING ******************
local dqn_save_path = "./dqn_model"
local dqn = func_create_dqn()
dqn = dqn:cuda()

-- training setting for dqn
local dqn_params, dqn_gradParams = dqn:getParameters()
local dqn_criterion = nn.SmoothL1Criterion():cuda()
local dqn_optimState = {learningRate = opt.lr, maxIteration = 1, learningRateDecay = opt.lrd, evalCounter = 0}

local dqn_logger = optim.Logger(opt.dqn_log)

local replay_memory = {}
local replay_memory_buffer_size = opt.replay_buffer
local gamma = 0.90 --discount factor
local epsilon = 1 -- greedy policy
local max_epochs = opt.epochs
local batch_size = opt.batch_size
local max_steps = opt.max_steps
local thd = 0.5
local lower_thd = 0.3
local sample_thd = 0.5 -- threshold for get samples to finetuning resnet_model
local action1_alpha = 0.55
local action2_alpha = 0.25
local train_period = 10
local count_train = torch.Tensor(1):fill(0)

-- local history_vector_size
-- local input_vector_size

--###########################################################

--********************************************************
-- ******************** RGN PREPARING ******************
local rgn_save_path = "./rgn_model"
local rgn = func_create_rgn()
rgn = rgn:cuda()
local rgn_logger = optim.Logger(opt.rgn_log)
local rgn_buffer = {}
local rgn_buffer_size = 64
-- training setting for rgn
local rgn_params, rgn_gradParams = rgn:getParameters()
local rgn_criterion = nn.AbsCriterion():cuda()
local rgn_optimState = {learningRate = 1e-3, maxIteration = 1, learningRateDecay = 0.0009, evalCounter = 0}

--###########################################################

--###########################################################

--********************************************************
-- ******************** PVN PREPARING ******************
local pvn_save_path = "./pvn_model"
local pvn = func_create_pvn()
pvn = pvn:cuda()
local pvn_logger = optim.Logger(opt.pvn_log)

-- training setting for pvn
local pvn_params, pvn_gradParams = pvn:getParameters()
local pvn_criterion = nn.CrossEntropyCriterion():cuda()
local pvn_optimState = {learningRate = 1e-3, maxIteration = 1, learningRateDecay = 0.0009, evalCounter = 0}

--###########################################################


--********************************************************
-- ********************** RUN DQN ************************

for i = 1,max_epochs do
	print('It is the ' .. i .. 'th epoch')
	local data_loop 
	if opt.load_saved_data == 0 then 
		data_loop = 1
	else
		data_loop = 4
	end

	for n = 1,data_loop do
		local data_st = os.clock()
		if opt.load_saved_data == 1 then
			images = nil
			annotations = nil
			local train_data = torch.load(dataset_path .. 'images' .. n ..'.t7')
			images = train_data.images
			train_data = torch.load(dataset_path .. 'annotations'.. n ..'.t7')
			annotations = train_data.annotations
			print('Loading Training data ' .. n ..' finished..')
		end

		for j,v in pairs(images) do
			local img_st = os.clock()
			print('\tIt is the ' .. j .. ' image')

			local cur_annotation = annotations[j]
			local gt_num = #cur_annotation
			local cur_img = v
			cur_img = func_image_processing_for_resnet(cur_img)
			local cur_img_size = {cur_img:size(2), cur_img:size(3)}

			local node_queue = {}
			local tmp_node = tree_node()

			
			tmp_node.detected_obj_table = torch.Tensor(gt_num):fill(0)


			tmp_node.cur_mask = {0.1,0.1, cur_img:size(2), cur_img:size(3)}

			-- iou_table record the iou of each gt and cur_mask
			-- reset iou_table in the beginning of each loop
			tmp_node.iou_table = torch.Tensor(gt_num):fill(0)
			tmp_node.old_iou = 0
			tmp_node.new_iou = 0
			local index -- index to the object corresponding to current iou

			-- calculate iou for cur_mask and gt
			tmp_node.old_iou, tmp_node.new_iou, tmp_node.iou_table, index, tmp_node.detected_obj_table = 
						func_follow_iou(tmp_node.cur_mask,cur_annotation,tmp_node.detected_obj_table,
										tmp_node.iou_table, thd)

			local  now_target_gt = cur_annotation[index]

			-- init feature input
			local conv_fea, class_softmax = func_get_image_conv_feature_and_softmax(
																		resnet_model, cur_img, 
																		tmp_node.cur_mask)
			
			local topk = 0
			if opt.enable_CAM == 1 then
				conv_fea = func_conv_cam_fusion(conv_fea, fc_weight, class_softmax, now_target_gt[1], topk,softmax_net)
			end
			conv_fea = conv_fea:view(conv_fea:nElement())
			local history_vector = torch.Tensor(history_vector_size):fill(0):cuda()
			local detected_obj_table = tmp_node.detected_obj_table:clone()
			tmp_node.history_vector = torch.Tensor(history_vector_size):fill(0):cuda()
			tmp_node.conv_fea = conv_fea
			tmp_node.input_vector = torch.cat(tmp_node.conv_fea, tmp_node.history_vector, 1)

			table.insert(node_queue, tmp_node)

			print('\tInitialization finished. Target GT = [' .. now_target_gt[2] .. ', ' .. now_target_gt[3] ..
					', ' .. now_target_gt[4] .. ', ' .. now_target_gt[5] .. ' ], ' .. 'Cur_IOU = ' .. 
					tmp_node.new_iou .. '; Detected_OBJ = ' .. tmp_node.detected_obj_table:sum() .. '\n')


			for k=1, max_steps do
				
				print('\t\tStep ' .. k .. ':\n')
				local reward1 = 0
				local reward2 = 0
				local cur_node = node_queue[1]

				local action_output = dqn:forward(cur_node.input_vector)
				--print(action_output)
				-- scaling action
				local tmp_v, action1 = torch.max(action_output[{{1,5}}],1)
				action1 = action1[1]-- from tensor to numeric type
				-- translation action
				local tmp_v, action2 = torch.max(action_output[{{6,number_of_actions}}],1)
				action2 = action2[1] + 5 -- from tensor to numeric type

				local rand_flag = 0
				if torch.uniform(torch.Generator()) < epsilon then -- greedy policy
					action1 = torch.random(torch.Generator(),1,5)
					action2 = torch.random(torch.Generator(),6,number_of_actions)
					rand_flag = 1
				end
				--print('\t\tcurrent mask:')
				--print(cur_node.cur_mask)
				print('\t\t\tTake scaling ' .. action1 .. ' and traslation ' .. action2 .. 
					'. rand_flag = ' .. rand_flag .. '\n')

				
				local new_node1 = tree_node()
				local now_target_gt1
				local class_softmax1
				local flag1
				local new_node2 = tree_node()
				local now_target_gt2
				local class_softmax2
				local flag2
				-- take action1
				--print('##take scaling action')

				new_node1, reward1, now_target_gt1,flag1, history_vector = func_run_dqn_action(
														new_node1, cur_node, action1, 
														action1_alpha, cur_annotation, 
														thd,cur_img_size,history_vector,detected_obj_table)
				--update detected_obj_table
				detected_obj_table = new_node1.detected_obj_table:clone()
				-- take action2
				--print('##take traslation action')
				new_node2, reward2, now_target_gt2,flag2, history_vector = func_run_dqn_action(
														new_node2, cur_node, action2,
														action2_alpha, cur_annotation, 
														thd, cur_img_size, history_vector,detected_obj_table)
				--update detected_obj_table
				detected_obj_table = new_node2.detected_obj_table:clone()
				if flag1 < 2 and flag2 < 2 then
					new_node1.conv_fea, class_softmax1, new_node2.conv_fea, class_softmax2 =
													func_get_2_conv_feature_and_softmax(
																resnet_model, cur_img, 
																new_node1.cur_mask,
																new_node2.cur_mask)
				   
					if opt.enable_CAM == 1 then
						new_node1.conv_fea = func_conv_cam_fusion(
												new_node1.conv_fea, fc_weight, 
												class_softmax1, now_target_gt1[1], topk,softmax_net)
						new_node2.conv_fea = func_conv_cam_fusion(
												new_node2.conv_fea, fc_weight, class_softmax2, 
												now_target_gt2[1], topk,softmax_net)
					end
					new_node1.conv_fea = new_node1.conv_fea:view(new_node1.conv_fea:nElement())
					new_node2.conv_fea = new_node2.conv_fea:view(new_node2.conv_fea:nElement())
					new_node1.input_vector = torch.cat(new_node1.conv_fea, new_node1.history_vector, 1)
					new_node2.input_vector = torch.cat(new_node2.conv_fea, new_node2.history_vector, 1)
				elseif flag1 < 2 then
					new_node1.conv_fea, class_softmax1 = func_get_image_conv_feature_and_softmax(
														resnet_model, cur_img,new_node1.cur_mask)
					if opt.enable_CAM == 1 then
						new_node1.conv_fea = func_conv_cam_fusion(
												new_node1.conv_fea, fc_weight, 
												class_softmax1, now_target_gt1[1], topk,softmax_net)
					end
					new_node1.conv_fea = new_node1.conv_fea:view(new_node1.conv_fea:nElement())
					new_node1.input_vector = torch.cat(new_node1.conv_fea, new_node1.history_vector, 1)
				elseif flag2 < 2 then
					new_node2.conv_fea, class_softmax2 = func_get_image_conv_feature_and_softmax(
														resnet_model, cur_img,new_node2.cur_mask)
					if opt.enable_CAM == 1 then
						new_node2.conv_fea = func_conv_cam_fusion(
												new_node2.conv_fea, fc_weight, class_softmax2, 
												now_target_gt2[1], topk,softmax_net)
					end
					new_node2.conv_fea = new_node2.conv_fea:view(new_node2.conv_fea:nElement())
					new_node2.input_vector = torch.cat(new_node2.conv_fea, new_node2.history_vector, 1)
				end


				--new_node1.conv_fea, class_softmax = func_get_image_conv_feature_and_softmax(
				--										resnet_model, cur_img,new_node1.cur_mask)
				
				if new_node1.new_iou > thd and flag1 < 2 then
					if opt.finetune_resnet == 1 then
						table.insert(resnet_buffer, 
							{func_image_preprocessing(cur_img, new_node1.cur_mask),
							 now_target_gt1[1]})
					end
					table.insert(rgn_buffer,{new_node1.conv_fea, 
						 func_gt_loc_reg(now_target_gt1, new_node1.cur_mask)})
				end

				if new_node2.new_iou > thd and flag2 < 2 then
					if opt.finetune_resnet == 1 then
						table.insert(resnet_buffer, 
							{func_image_preprocessing(cur_img, new_node2.cur_mask),
							 now_target_gt2[1]})
					end
					table.insert(rgn_buffer,{new_node2.conv_fea, 
						 func_gt_loc_reg(now_target_gt2, new_node2.cur_mask)})
				end
				
				
				--print('##take traslation action over\n\n')
				local obj_y_o_n = 0
				if cur_node.new_iou > thd then 
					obj_y_o_n = 1 
				elseif cur_node.new_iou < lower_thd then
					obj_y_o_n = -1 
				end
					-- insert background cur_node img if #resnet_buffer > resnet_batch_size -3
					if opt.enable_background == 1 and opt.finetune_resnet == 1 then
						if #resnet_buffer > resnet_batch_size -2 then
							table.insert(resnet_buffer, 
								{func_image_preprocessing(cur_img, cur_node.cur_mask),21})
						end

					end

				
				local tmp_experience = {cur_node.input_vector, action1, action2,
										reward1, reward2, new_node1.input_vector,
										new_node2.input_vector, obj_y_o_n}

				if #replay_memory < replay_memory_buffer_size then
					table.insert(replay_memory, tmp_experience)
				else
					table.remove(replay_memory, 1)
					table.insert(replay_memory, tmp_experience)
				end
				
				--***********************************************************
				--*****************Training DQN & PVN******************************
				local tmp_mod = torch.fmod(count_train,train_period)
				tmp_mod = tmp_mod[1]
				if #replay_memory > replay_memory_buffer_size/2  then
					if tmp_mod == 0 then
						local st = os.clock()
						count_train[1] = count_train[1]+1
						local minibatch = func_sample(replay_memory, batch_size)
						local  et = os.clock()
						--print('sample minibatch time = ' .. et-st)
						local training_set = {data=torch.Tensor(batch_size, input_vector_size):cuda(),
												label=torch.Tensor(batch_size, number_of_actions):cuda()}

						local training_set_pvn = {data=torch.Tensor(batch_size, feature_size):cuda(),
												label=torch.Tensor(batch_size, 1):fill(0):cuda()}
					
						if i < 3 then
							training_set = func_construct_dqn_training_data(minibatch, training_set, dqn, gamma)
						else

							training_set, training_set_pvn = func_construct_dqn_pvn_training_data(
																minibatch, training_set, training_set_pvn, dqn, gamma)
												
						end
						st = os.clock()
						--print('construct data time = ' .. st - et)
						print('\t\t\t\t Training DQN...\n')

						local function feval(x)
							if x ~= dqn_params then
								dqn_params:copy(x)
							end
							dqn_gradParams:zero()
							local outputs = dqn:forward(training_set.data)
							local loss = dqn_criterion:forward(outputs, training_set.label)
							local dloss_doutputs = dqn_criterion:backward(outputs, training_set.label)
							dqn:backward(training_set.data, dloss_doutputs)
							dqn_logger:add{loss}
							return loss, dqn_gradParams 
						end
						optim.sgd(feval, dqn_params, dqn_optimState)
						if dqn_optimState.learningRateDecay > 0 and 
							dqn_optimState.learningRate / (1+dqn_optimState.learningRateDecay*dqn_optimState.evalCounter) < 0.0001 then
							dqn_optimState.learningRate = 0.0001
							dqn_optimState.learningRateDecay = 0

						end
						et = os.clock()
						--print('training time = ' .. et -st)
						local function feval_pvn(x)
							if x ~= pvn_params then
								pvn_params:copy(x)
							end
							pvn_gradParams:zero()
							local outputs = pvn:forward(training_set_pvn.data)
							local loss = pvn_criterion:forward(outputs, training_set_pvn.label)
							local dloss_doutputs = pvn_criterion:backward(outputs, training_set_pvn.label)
							pvn:backward(training_set_pvn.data, dloss_doutputs)
							pvn_logger:add{loss}
							return loss, pvn_gradParams 
						end
						if i > 2 then
							optim.sgd(feval_pvn, pvn_params, pvn_optimState)
						end
					else
						count_train[1] = count_train[1]+1
					end
				end
				--#############################################################

				--*************************************************************
				--********************Training resnet and rgn*****************
				if #rgn_buffer >= rgn_buffer_size then
					print('training rgn')
					-- Training rgn first
					-- feature_size is global variable
					local training_set = {data=torch.Tensor(#rgn_buffer, feature_size):cuda(),
											label=torch.Tensor(#rgn_buffer, 4):cuda()}
					training_set = 	func_construct_rgn_training_data(rgn_buffer,training_set)
					local function rgn_feval(x)
						if x ~=  rgn_params then
							rgn_params:copy(x)
						end
						rgn_gradParams:zero()
						local outputs = rgn:forward(training_set.data)
						local loss = rgn_criterion:forward(outputs, training_set.label)
						local dloss_doutputs = rgn_criterion:backward(outputs, training_set.label)
						rgn:backward(training_set.data, dloss_doutputs)
						rgn_logger:add{loss}
						return loss, rgn_gradParams 
					end
					optim.sgd(rgn_feval, rgn_params, rgn_optimState)
					rgn_buffer = {}
				end

				if #resnet_buffer >= resnet_batch_size then	

					-- Finetunning resnet
					if opt.finetune_resnet == 1 then
						print('Finetunning resnet')
						local training_set = {data=torch.Tensor(#resnet_buffer, resnet_buffer[1][1]:size(2),
							resnet_buffer[1][1]:size(3),resnet_buffer[1][1]:size(4)):cuda(),
											label=torch.Tensor(#resnet_buffer, 1):cuda()}
						training_set = 	func_construct_resnet_training_data(resnet_buffer,training_set)
						local function resnet_feval(x)
							if x ~= resnet_params then
								resnet_params:copy(x)
							end
							resnet_gradParams:zero()
							local outputs = resnet_model:forward(training_set.data)
							local loss = resnet_criterion:forward(outputs, training_set.label)
							local dloss_doutputs = resnet_criterion:backward(outputs, training_set.label)
							resnet_model:backward(training_set.data, dloss_doutputs)
							resnet_logger:add{loss}
							return loss, resnet_gradParams 
						end
						optim.sgd(resnet_feval, resnet_params, resnet_optimState)
					end

					-- flush the buffer
					resnet_buffer={}
				end

				table.insert(node_queue, new_node1)
				table.insert(node_queue, new_node2)
				table.remove(node_queue, 1)
				
			end -- step loop
			local img_et = os.clock()
			--print('image time = ' .. img_et - img_st)
		end -- image loop
		local data_et = os.clock()
		print('data loop time = '.. (data_et - data_st)/3600 .. ' h')
	end -- data loop
	if epsilon > 0.1 then 
		epsilon = epsilon - 0.15
	end

		-- save models
	local model_name = './model/dqn/' .. opt.name .. '_' .. i .. '.t7'
	torch.save(model_name, {dqn = dqn})
	model_name = './model/rgn/' .. opt.name .. '_' .. i .. '.t7'
	torch.save(model_name, {rgn = rgn})
	model_name = './model/pvn/' .. opt.name .. '_' .. i .. '.t7'
	torch.save(model_name, {pvn = pvn})
	if opt.finetune_resnet == 1 then
		model_name = './model/resnet/' .. opt.name .. '_' .. i .. '.t7'
		torch.save(model_name, {resnet_model = resnet_model})
	end
		-- evaluate model
		--[[
	if opt.enable_evaluate == 1 then
	
	end
	--]]

end -- epoch loop