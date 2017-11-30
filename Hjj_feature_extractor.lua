require 'image'
require 'torch'
-- for image processing
local t = require 'fb.resnet.torch-master/datasets/transforms'

function func_image_processing_for_resnet(img)
	-- The resnet was trained with this input normalization
	local meanstd = {
   		mean = { 0.485, 0.456, 0.406 },
   		std = { 0.229, 0.224, 0.225 },
	}

	local transform = t.Compose{
   		--t.Scale(256),
   		t.ColorNormalize(meanstd),
   		--t.CenterCrop(224),
	}

	img = transform(img)

	return img

end

function func_image_preprocessing( img, bbx )
--####################################
-- crop image and do preprocessing
-- return image vector for resnet
--#####################################
	--print(img:size())
	--print(bbx)
	local crop_img = image.crop(img, bbx[2], bbx[1], bbx[4], bbx[3])
	crop_img = image.scale(crop_img,224,224)
	return crop_img:view(1, table.unpack(crop_img:size():totable())):cuda()

end

function func_get_batch_conv_feature_and_softmax(resnet_model, cur_img, node_queue)
	local img_vector1 = func_image_preprocessing(cur_img, node_queue[1].cur_mask)
	local img_vector2 = func_image_preprocessing(cur_img, node_queue[2].cur_mask)
	local num = #node_queue
	for i=3,num,2 do
		img_vector1 = torch.cat(img_vector1,func_image_preprocessing(cur_img, node_queue[i].cur_mask),1)
	end

	for i=4,num,2 do
		img_vector2 = torch.cat(img_vector2,func_image_preprocessing(cur_img, node_queue[i].cur_mask),1)
	end

	local input_vec = torch.cat(img_vector1,img_vector2,1)
	local class_softmax = resnet_model:forward(input_vec:cuda())
	--print(class_softmax:size())
	return resnet_model:get(9).output[{{1,num/2}}], class_softmax[{{1,num/2}}],
			resnet_model:get(9).output[{{num/2+1,num}}], class_softmax[{{num/2+1,num}}]			
end

function func_get_2_conv_feature_and_softmax(resnet_model, cur_img,cur_mask1,cur_mask2 )
	local img_vector1 = func_image_preprocessing(cur_img, cur_mask1)
	local img_vector2 = func_image_preprocessing(cur_img, cur_mask2)
	--print(img_vector1:size())
	local input_vec = torch.cat(img_vector1,img_vector2,1)
	local class_softmax = resnet_model:forward(input_vec:cuda())
	--print(class_softmax:size())
	return resnet_model:get(9).output[1], class_softmax[1],resnet_model:get(9).output[2], class_softmax[2]
end




function func_get_image_conv_feature_and_softmax(resnet_model, cur_img,cur_mask)
--####################################
-- get the conv feature of the part covered by cur_mask
-- return conv feature vector and softmax (cuda)
--#####################################
	local cur_img_vector = func_image_preprocessing(cur_img, cur_mask)
	local class_softmax = resnet_model:forward(cur_img_vector)
	return resnet_model:get(9).output, class_softmax
end


function func_conv_cam_fusion(conv_fea, fc_weight, class_softmax, now_target_class, topk,softmax_net)
	local idx_table={}
	if topk == 0 then
		-- use the ground truth result
		table.insert(idx_table, now_target_class)
	else
		local v, idx = torch.sort(-class_softmax)
		for i=1,topk do table.insert(idx_table, idx[i]) end
	end

	conv_fea = torch.squeeze(conv_fea)
	--print(conv_fea:size())
	local fea_map = torch.Tensor(conv_fea:size(2), conv_fea:size(3)):fill(0):cuda()
	--print(fc_weight:size())
	for i, v in pairs(idx_table) do
		local tmp_fea_map = conv_fea:clone():cuda()
		--[[
		for k=1,fc_weight:size(2) do
			tmp_fea_map[{{k},{},{}}] = tmp_fea_map[{{k},{},{}}] * fc_weight[v][k]
		end
		--]]
		tmp_fea_map:cmul(torch.expand(fc_weight[v]:view(128,1,1),128,7,7))
		--print(tmp_fea_map:size())
		tmp_fea_map = torch.sum(tmp_fea_map,1)
		tmp_fea_map = torch.squeeze(tmp_fea_map)
		--print(tmp_fea_map:size())
		--print(fea_map:size())
		--print(tmp_fea_map:size())
		fea_map  = fea_map + tmp_fea_map
	end
	
	fea_map = softmax_net:forward(fea_map:view(1,49))
	fea_map = fea_map:view(7,7)
	--fea_map = (fea_map - torch.min(fea_map))/(torch.max(fea_map)-torch.min(fea_map))
	--fea_map = fea_map + 0.1
	
	--[[
	for i=1, conv_fea:size(1) do
		conv_fea[{{i},{},{}}]:cmul(fea_map) 
	end
	--]]
	conv_fea:cmul(torch.expand(fea_map:view(1,7,7),128,7,7))
	
	return conv_fea
end

function func_get_cam(conv_fea, fc_weight,class_softmax, softmax_net, img_size)
	local idx_table = {}
	local cam_table = {}
	local topk = 5
	local v, idx = torch.sort(-class_softmax)
	for i=1,topk do table.insert(idx_table, idx[i]) end

	conv_fea = torch.squeeze(conv_fea)
	--print(conv_fea:size())
	local fea_map = torch.Tensor(conv_fea:size(2), conv_fea:size(3)):fill(0):cuda()

	for i, v in pairs(idx_table) do
		local tmp_fea_map = conv_fea:clone():cuda()
		tmp_fea_map:cmul(torch.expand(fc_weight[v]:view(128,1,1),128,7,7))
		--print(tmp_fea_map:size())
		tmp_fea_map = torch.sum(tmp_fea_map,1)
		tmp_fea_map = torch.squeeze(tmp_fea_map)

		fea_map  = fea_map + tmp_fea_map

		if i == 1 or i == 3 or i == 5 then
			local t_fea_map = softmax_net:forward(fea_map:view(1,49))
			t_fea_map = t_fea_map:view(7,7)
			--t_fea_map  = image.scale(t_fea_map ,img_size[2], img_size[1])
			table.insert(cam_table, t_fea_map:float())
		end
	end
	return cam_table

end