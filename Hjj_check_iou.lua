function func_gt_loc_reg(gt, mask)

	local gt_mask = {gt[2],gt[3],gt[4],gt[5]}
	local h = mask[3] - mask[1]
	local w = mask[4] - mask[2]
	return torch.Tensor{(gt_mask[1] - mask[1])/h,
						(gt_mask[2] - mask[2])/w,
						(gt_mask[3] - mask[3])/h,
						(gt_mask[4] - mask[4])/w}:cuda()

end

--[[
function func_calculate_iou( mask, gt )
	--print('calculate iou:')
	--print(mask)
	local gt_mask = {math.ceil(gt[2]), math.ceil(gt[3])
					,math.floor(gt[4]),math.floor(gt[5])}
	mask[1] = math.ceil(mask[1])
	mask[2] = math.ceil(mask[2])
	if mask[1] <= 0 then mask[1] = 1 end
	if mask[2] <= 0 then mask[2] = 1 end
	mask[3] = math.floor(mask[3])
	if mask[3] < mask[1] then mask[3] = mask[1] end
	mask[4] = math.floor(mask[4])
	if mask[4] < mask[2] then mask[4] = mask[2] end

	local map_size = {}

	if gt_mask[3] >= mask[3] then
		table.insert(map_size,gt_mask[3])
	else
		table.insert(map_size, mask[3])
	end

	if gt_mask[4] >= mask[4] then
		table.insert(map_size,gt_mask[4])
	else
		table.insert(map_size,mask[4])
	end

	local gt_map = torch.LongTensor(map_size[1], map_size[2]):fill(0)
	gt_map[{ {gt_mask[1], gt_mask[3]}, {gt_mask[2], gt_mask[4]}}]:fill(1)

	local mask_map = torch.LongTensor(map_size[1], map_size[2]):fill(0)
	--print(mask_map:size())
	--print(mask)
	mask_map[{{mask[1], mask[3]}, {mask[2], mask[4]}}]:fill(1)

	return torch.cbitand(gt_map, mask_map):sum() / torch.cbitor(gt_map, mask_map):sum()

end
--]]


function func_calculate_iou( mask, gt )
	local gt_mask = {gt[3],gt[2],gt[5],gt[4]}
	local m_area = (mask[3]-mask[1])*(mask[4]-mask[2])
	local g_area = (gt_mask[3]-gt_mask[1])*(gt_mask[4]-gt_mask[2])
	local x1 = math.max(mask[1],gt_mask[1])
	local y1 = math.max(mask[2], gt_mask[2])
	local x2 = math.min(mask[3],gt_mask[3])
	local y2 = math.min(mask[4],gt_mask[4])
	local w = math.max(0,x2-x1)
	local h = math.max(0,y2-y1)
	return (w*h)/(m_area+g_area-(w*h))

end

function func_follow_iou(mask, gt, detected_obj_table, iou_table, thd)
	local result_table = torch.Tensor(iou_table:size()):fill(0)
	local iou = 0
	local new_iou = 0
	local index = 0
	local new_detected_obj_table = detected_obj_table:clone()

	for i = 1, #gt do 
		iou = func_calculate_iou(mask, gt[i])
		result_table[i] = iou
		if iou >= thd then new_detected_obj_table[i] = 1 end
	end

	new_iou, index = torch.max(result_table, 1)
	-- from tensor to numeric type
	new_iou = new_iou[1]
	index = index[1]

	iou = iou_table[index]

	return iou, new_iou, result_table, index, new_detected_obj_table
end



function func_get_reward(old_iou, new_iou, new_detected_obj_table, old_detected_obj_table)
	local reward
	if new_iou - old_iou > 0.007 then
		reward = 5
	else
		reward = -1
	end

	if (new_detected_obj_table - old_detected_obj_table):sum() > 0 then
		reward = 10 * (new_detected_obj_table - old_detected_obj_table):sum()
	end

	return reward

end
