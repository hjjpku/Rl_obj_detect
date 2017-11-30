require 'Hjj_check_iou'
require 'Hjj_reinforcement'

function func_take_action(img_size, cur_mask, action, action_alpha)
-- img_size -> {x_max, y_max}
--#### 1~5 scaling actions ; 6~13 trasnlation actions
-- 1 -> top left
-- 2 -> top right
-- 3 -> bottom left
-- 4 -> bottom right
-- 5 -> middle
-- 6 -> move left
-- 7 -> move right
-- 8 -> move top
-- 9 -> move bottom
-- 10 -> height shrink
-- 11 -> width shrink
-- 12 -> height expand
-- 13 -> width expand

-- return new_mask

	local new_mask = {}
	local offset


	if action == 1 then
		table.insert(new_mask, cur_mask[1])
		table.insert(new_mask, cur_mask[2])
		table.insert(new_mask, cur_mask[1] + (cur_mask[3]-cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[2] + (cur_mask[4] - cur_mask[2])*action_alpha)
	elseif action == 2 then
		table.insert(new_mask, cur_mask[1])
		table.insert(new_mask, cur_mask[4] - (cur_mask[4] - cur_mask[2])*action_alpha)
		table.insert(new_mask, cur_mask[1] + (cur_mask[3] - cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[4])
	elseif action == 3 then
		table.insert(new_mask, cur_mask[3] - (cur_mask[3] - cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[2])
		table.insert(new_mask, cur_mask[3])
		table.insert(new_mask, cur_mask[2] + (cur_mask[4] - cur_mask[2])*action_alpha)
	elseif action == 4 then
		table.insert(new_mask, cur_mask[3] - (cur_mask[3] - cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[4] - (cur_mask[4] - cur_mask[2])*action_alpha)
		table.insert(new_mask, cur_mask[3])
		table.insert(new_mask, cur_mask[4])
	elseif action == 5 then
		table.insert(new_mask, cur_mask[1] + (cur_mask[3] - cur_mask[1])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[2] + (cur_mask[4] - cur_mask[2])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[3] - (cur_mask[3] - cur_mask[1])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[4] - (cur_mask[4] - cur_mask[2])*(1-action_alpha)/2)		
	elseif action == 6 then	
		table.insert(new_mask, cur_mask[1])
		offset = cur_mask[2] - (cur_mask[4]-cur_mask[2])*action_alpha
		if offset > 0 then
			table.insert(new_mask, offset)
		else 
			table.insert(new_mask, 0.1)
		end
		table.insert(new_mask, cur_mask[3])
		offset = cur_mask[4] - (cur_mask[4]-cur_mask[2])*action_alpha
		table.insert(new_mask, offset)

		--table.insert(new_mask, new_mask[2] + (cur_mask[4]-cur_mask[2]))
	elseif action == 7 then
		table.insert(new_mask, cur_mask[1])
		table.insert(new_mask, cur_mask[2]+ (cur_mask[4] - cur_mask[2])*action_alpha)
		table.insert(new_mask, cur_mask[3])
		offset = cur_mask[4] + (cur_mask[4] - cur_mask[2])*action_alpha
		if offset > img_size[2] then
			table.insert(new_mask, img_size[2])
			--new_mask[2] = new_mask[4] - (cur_mask[4] - cur_mask[2])
		else
			table.insert(new_mask, offset)
		end		
	elseif action == 8 then
		offset = cur_mask[1] - (cur_mask[3] - cur_mask[1])*action_alpha
		if offset > 0 then
			table.insert(new_mask, offset)
		else
			table.insert(new_mask, 0.1)
		end
		table.insert(new_mask, cur_mask[2])
		table.insert(new_mask, cur_mask[3] - (cur_mask[3] - cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[4])
	elseif action == 9 then	
		table.insert(new_mask, cur_mask[1] + (cur_mask[3] - cur_mask[1])*action_alpha)
		table.insert(new_mask, cur_mask[2])
		offset = cur_mask[3] + (cur_mask[3] - cur_mask[1])*action_alpha
		if offset > img_size[1] then
			table.insert(new_mask, img_size[1])
			--new_mask[1] = new_mask[3] - (cur_mask[3] - cur_mask[1])
		else 
			table.insert(new_mask, offset)
		end
		table.insert(new_mask, cur_mask[4])
	elseif action == 10 then
		table.insert(new_mask, cur_mask[1] + (cur_mask[3] - cur_mask[1])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[2])
		table.insert(new_mask, cur_mask[3] - (cur_mask[3] - cur_mask[1])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[4])
		
	elseif action == 11 then
		table.insert(new_mask, cur_mask[1])
		table.insert(new_mask, cur_mask[2] + (cur_mask[4] - cur_mask[2])*(1-action_alpha)/2)
		table.insert(new_mask, cur_mask[3])
		table.insert(new_mask, cur_mask[4] - (cur_mask[4] - cur_mask[2])*(1-action_alpha)/2)
		
	elseif action == 12 then
		offset = cur_mask[1] - (cur_mask[3] - cur_mask[1])*action_alpha/2
		if offset > 0 then
			table.insert(new_mask, offset)
		else
			table.insert(new_mask, 0.1)
		end
		table.insert(new_mask, cur_mask[2])
		offset = cur_mask[3] + (cur_mask[3] - cur_mask[1])*action_alpha/2
		if offset <= img_size[1] then
			table.insert(new_mask, offset)
		else
			table.insert(new_mask, img_size[1])
		end
		table.insert(new_mask, cur_mask[4])
	elseif action == 13 then
		table.insert(new_mask, cur_mask[1])
		offset = cur_mask[2] - (cur_mask[4] - cur_mask[2])*action_alpha/2
		if offset > 0 then
			table.insert(new_mask, offset)
		else
			table.insert(new_mask, 0.1)
		end
		table.insert(new_mask, cur_mask[3])
		offset = cur_mask[4] + (cur_mask[4] - cur_mask[2])*action_alpha/2
		if offset <= img_size[2] then
			table.insert(new_mask, offset)
		else
			table.insert(new_mask, img_size[2])
		end
	else
		print('error action')
		os.exit()
	end	

	--do not shrink, keep at least 4 pixel
	local flag = 0
	if new_mask[3] - new_mask[1] < 7 then 
		new_mask[3] = cur_mask[3]
		new_mask[1] = cur_mask[1]
		flag = flag + 1
	end
	if new_mask[4] - new_mask[2] < 7 then 
		new_mask[4] = cur_mask[4]
		new_mask[2] = cur_mask[2]
		flag = flag+1
	end
	return new_mask,flag

end

function func_update_history_vector(history_vector, action)
	local updated_history_vector = torch.Tensor(number_of_actions*history_action_buffer_size):zero():cuda()
	if history_action_buffer_size == 0 then
		return updated_history_vector
	end
	local stored_action_number = history_vector:nonzero():numel()
	--print('stored_action_number = ' .. stored_action_number)
	if history_action_buffer_size > stored_action_number then
		updated_history_vector = history_vector:clone()
		updated_history_vector[stored_action_number*number_of_actions + action] = 1 
	else
		updated_history_vector[{ {1,number_of_actions*(history_action_buffer_size-1)} }] = 
			history_vector[{ {number_of_actions+1, -1} }]
		updated_history_vector[-(number_of_actions-action+1)] = 1
	end	
	
	return updated_history_vector
end


function func_run_dqn_action( new_node, cur_node, action, action_alpha, cur_annotation, thd, img_size, history_vector, detected_obj_table)
	local index
	local flag
	new_node.cur_mask, flag = func_take_action(img_size, cur_node.cur_mask, action, action_alpha)
	if flag == 2 then
		new_node = cur_node
	end
	
	new_node.old_iou, new_node.new_iou, new_node.iou_table, index, new_node.detected_obj_table = 
			func_follow_iou(new_node.cur_mask,cur_annotation,detected_obj_table,
							cur_node.iou_table, thd)

	now_target_gt = cur_annotation[index]

	local reward = func_get_reward(new_node.old_iou, new_node.new_iou, new_node.detected_obj_table,
							detected_obj_table)
	--local reward = func_get_reward(cur_node.iou_table, new_node.iou_table, new_node.detected_obj_table,
	--						detected_obj_table)

	if reward > 0 and (action == 13 or action == 12) then
		reward = 2
	end

	history_vector = func_update_history_vector(history_vector, action)
	new_node.history_vector = history_vector:clone()

	print('\t\t\tAfter action ' .. action .. ' :\n')
	print('\t\t\t\tReward = ' .. reward .. '; Cur_IOU = ' .. 
			new_node.new_iou .. '; Detected_OBJ = ' .. new_node.detected_obj_table:sum() .. '\n')
	--[[
	print('\t\t\tTarget GT = [' .. now_target_gt[2] .. ', ' .. now_target_gt[3] ..
			', ' .. now_target_gt[4] .. ', ' .. now_target_gt[5] .. ' ], ' .. 'Cur_IOU = ' .. 
			new_node.new_iou .. '; Detected_OBJ = ' .. new_node.detected_obj_table:sum() .. '\n')
	--]]
	--print('\t\t\tnew mask:')
	--print(new_node.cur_mask)
	return new_node, reward, now_target_gt, flag, history_vector,index
end


function func_localization_regression(loc_vec, cur_mask,img_size)
	local mask = {1,1,img_size[1],img_size[2]}
	local h = cur_mask[3] - cur_mask[1]
	local w = cur_mask[4] - cur_mask[2]
	local loc = loc_vec[1]*h+cur_mask[1]
	mask[1] = math.max(mask[1], loc)
	loc = loc_vec[2]*w+cur_mask[2]
	mask[2] = math.max(mask[2], loc)
	loc = loc_vec[3]*h + cur_mask[3]
	mask[3] = math.min(loc, mask[3])
	loc = loc_vec[4]*w+cur_mask[4]
	mask[4] = math.min(loc, mask[4])
	if mask[3] <= mask[1] or mask[4] <=  mask[2] then
		print('loc reg failed!')
		return cur_mask
	else
		return mask
	end

end