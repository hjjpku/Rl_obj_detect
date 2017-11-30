--#########################################
-- load images from VOC
--
--##########################################

require 'image'
require 'torch'
require 'io'
require 'LuaXML'

-- for image processing
local t = require 'fb.resnet.torch-master/datasets/transforms'

function func_load_image_names(data_set_name,path)
--####################################
-- load image names
-- return image names table
--#####################################
	local image_names = {}
	file_path = path .. '/ImageSets/Main/' .. data_set_name .. '.txt'
	for line in io.lines(file_path) do
		table.insert(image_names, line)
	end
	return image_names
end

function func_get_class_name( class_id)
	if class_id == 1 then
        return 'aeroplane'
    elseif class_id == 2 then
        return 'bicycle'
    elseif class_id == 3 then
        return 'bird'
    elseif class_id == 4 then
        return 'boat'
    elseif class_id == 5 then
        return 'bottle'
    elseif class_id == 6 then
        return 'bus'
    elseif class_id == 7 then
        return 'car'
    elseif class_id == 8 then
        return 'cat'
    elseif class_id == 9 then
        return 'chair'
    elseif class_id == 10 then
        return 'cow'
    elseif class_id == 11 then
        return 'diningtable'
    elseif class_id == 12 then
        return 'dog'
    elseif class_id == 13 then
        return 'horse'
    elseif class_id == 14 then
        return 'motorbike'
    elseif class_id == 15 then
        return 'person'
    elseif class_id == 16 then
        return 'pottedplant'
    elseif class_id == 17 then
        return 'sheep'
    elseif class_id == 18 then
        return 'sofa'
    elseif class_id == 19 then
        return 'train'
    elseif class_id == 20 then
        return 'tvmonitor'
    end
end

function func_get_class_id( class_name)
	if class_name == 'aeroplane' then
        return 1
    elseif class_name == 'bicycle' then
        return 2
    elseif class_name == 'bird' then
        return 3
    elseif class_name == 'boat' then
        return 4
    elseif class_name == 'bottle' then
        return 5
    elseif class_name == 'bus' then
        return 6
    elseif class_name == 'car' then
        return 7
    elseif class_name == 'cat' then
        return 8
    elseif class_name == 'chair' then
        return 9
    elseif class_name == 'cow' then
        return 10
    elseif class_name == 'diningtable' then
        return 11
    elseif class_name == 'dog' then
        return 12
    elseif class_name == 'horse' then
        return 13
    elseif class_name == 'motorbike' then
        return 14
    elseif class_name == 'person' then
        return 15
    elseif class_name == 'pottedplant' then
        return 16
    elseif class_name == 'sheep' then
        return 17
    elseif class_name == 'sofa' then
        return 18
    elseif class_name == 'train' then
        return 19
    elseif class_name == 'tvmonitor' then
        return 20
    end
end



function func_load_annotations( name, path )
--####################################
-- load image annotations
-- return image annotations table
-- 1 -> label; 2 -> xmin; 3 -> ymin; 4 -> xmax; 5 -> ymax
--#####################################
	local file_name = path .. '/Annotations/' .. name .. '.xml'
	local data = xml.load(file_name)
	local annotations = {}
	for i = 1, #data do
		local tmp_data = data[i]

		if tmp_data:find('name') and tmp_data:find('bndbox') then			
			local obj_class_name = tmp_data:find('name')[1] 
			local obj_class = func_get_class_id(obj_class_name)
			local obj_bbx = tmp_data:find('bndbox')
			table.insert(annotations, {obj_class, obj_bbx:find('xmin')[1],
				obj_bbx:find('ymin')[1], obj_bbx:find('xmax')[1], obj_bbx:find('ymax')[1]})
		end
	end
	return annotations
end



function func_image_loader(dataset, path_voc, path_voc2, cls)
--####################################
-- load images
-- return images table and annotations
--#####################################

	local image_names1 = {}
	local image_names2 = {}
	local image_names = {}

	if cls == nil then
		if dataset == 2 then
			image_names1 = func_load_image_names('trainval',path_voc)
			image_names2 = func_load_image_names('trainval',path_voc2)
			
            --image_names1 = func_load_image_names('1_aeroplane_trainval',path_voc)
            --image_names2 = func_load_image_names('1_aeroplane_trainval',path_voc2)
            image_names = {image_names1, image_names2}
		elseif dataset == 1 then
			image_names = {func_load_image_names('trainval',path_voc)}
		end
	else
		local cls_name = func_get_class_name(cls)
		image_names = {func_load_image_names('1_' .. cls_name .. '_test',path_voc)}
	end
	--print(image_names)

	local images = {}
	local annotations = {}

	for i, v in pairs(image_names) do
		if i==1  then
			for j, name in pairs(v) do
				local tmp_img = image.load(path_voc .. '/JPEGImages/' .. name .. '.jpg',3, 'float')
				--tmp_img = transform(tmp_img)
				table.insert(images, tmp_img)
				table.insert(annotations, func_load_annotations(name, path_voc))
			end
		else
			for j, name in pairs(v) do
				print(j)
				local tmp_img = image.load(path_voc2 .. '/JPEGImages/' .. name .. '.jpg',3, 'float')
				--tmp_img = transform(tmp_img)
				table.insert(images, tmp_img)
				table.insert(annotations, func_load_annotations(name, path_voc2))
			end
		end
	end
	return images, annotations, image_names
end


function func_write_prop_file(prop_file, tmp_record)
	prop_file:write(tmp_record[1] .. ' ' .. tmp_record[2] .. ' ' .. tmp_record[3][1] .. 
		' ' .. tmp_record[3][2] .. ' ' .. tmp_record[3][3] .. ' ' .. tmp_record[3][4] .. 
		' ' .. tmp_record[4] .. ' ' .. tmp_record[5] .. ' ' .. tmp_record[6] .. 
		' ' .. tmp_record[7] .. ' ' .. tmp_record[8] .. ' ' .. tmp_record[9] ..
		' ' .. tmp_record[10] .. '\n')
	return
end