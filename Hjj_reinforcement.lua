require 'cudnn'
--***************************************************************
--************************** FEATURE SIZE ***********************
history_action_buffer_size = 20
number_of_actions = 13
history_vector_size = number_of_actions * history_action_buffer_size
feature_size = 128 * 7 * 7
input_vector_size = feature_size + history_vector_size


--#################################################################

function func_create_pvn()
	local feature_dim = feature_size
	local hid_dim_1 = 4096
	local hid_dim_2 = 1024
	local output_dim = 2
	local softMaxLayer = cudnn.SoftMax()

	local net = nn.Sequential()
	net:add(nn.Reshape(feature_dim))
	net:add(nn.Linear(feature_dim, hid_dim_1))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_1, hid_dim_2))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_2, hid_dim_2))				
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_2, output_dim))
	net:add(softMaxLayer)
	return net
end


function func_create_dqn()
	local feature_dim = input_vector_size
	local hid_dim_1 = 4096
	local hid_dim_2 = 1024
	local output_dim = number_of_actions
	
	local net = nn.Sequential()
	net:add(nn.Reshape(feature_dim))
	net:add(nn.Linear(feature_dim, hid_dim_1))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim_1, hid_dim_2))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim_2, hid_dim_2))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim_2, hid_dim_2))				
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim_2, output_dim))
	return net
end


function func_create_rgn( )
	local feature_dim = feature_size
	local hid_dim_1 = 4096
	local hid_dim_2 = 1024
	local output_dim = 4

	local net = nn.Sequential()
	net:add(nn.Reshape(feature_dim))
	net:add(nn.Linear(feature_dim, hid_dim_1))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_1, hid_dim_2))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_2, hid_dim_2))				
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.5))
	net:add(nn.Linear(hid_dim_2, output_dim))
	return net
end

function func_sample(data, batch_size) 
	local minibatch = {}
	for i = 1, batch_size 
	do
		table.insert(minibatch,data[torch.random(
					torch.Generator(),1, table.getn(data))])
	end
	return minibatch
end
--[[
function func_construct_dqn_training_data( minibatch, training_set, dqn,gamma )
	for l, memory in pairs(minibatch) do
		local tmp_input_vector = memory[1]
		local tmp_action1 = memory[2]
		local tmp_action2 = memory[3]
		local tmp_reward1 = memory[4]
		local tmp_reward2 = memory[5]
		local tmp_new_input_vector1 = memory[6]
		local tmp_new_input_vector2 = memory[7]
		local old_action_output = dqn:forward(tmp_input_vector)
		local new_action_output1 = dqn:forward(tmp_new_input_vector1)
		local new_action_output2 = dqn:forward(tmp_new_input_vector2)
		local y = old_action_output:clone()
		local tmp_v, tmp_index = torch.max(new_action_output1,1)
		tmp_v = tmp_v[1]
		tmp_index = tmp_index[1]
		local update_reward = tmp_reward1 + gamma * tmp_v
		y[tmp_action1] = update_reward
		tmp_v, tmp_index = torch.max(new_action_output2,1)
		tmp_v = tmp_v[1]
		tmp_index = tmp_index[1]
		local update_reward = tmp_reward2 + gamma * tmp_v
		y[tmp_action2] = update_reward
		training_set.data[l] = tmp_input_vector
		training_set.label[l] = y
	end
	return training_set
end
--]]

function func_construct_dqn_training_data( minibatch, training_set, dqn,gamma )
	local batch_size = #minibatch
	local tmp_training_data = torch.Tensor(batch_size*3, input_vector_size):fill(0):cuda()
	local tmp_action1 = {}
	local tmp_action2 = {}
	local tmp_reward1 = torch.Tensor(batch_size, 1):fill(0):cuda()
	local tmp_reward2 = torch.Tensor(batch_size, 1):fill(0):cuda()

	for l,memory in pairs(minibatch) do
		tmp_training_data[l] = memory[1]
		--print(memory[6])
		tmp_training_data[l+batch_size] = memory[6]
		tmp_training_data[l+2*batch_size] = memory[7]

		table.insert(tmp_action1, memory[2])
		table.insert(tmp_action2, memory[3])

		tmp_reward1[l] = memory[4]
		tmp_reward2[l] = memory[5]
	end
	local output = dqn:forward(tmp_training_data)
	local output1 = output[{{1,batch_size}, {}}]:clone()
	local output2 = output[{{batch_size+1,2*batch_size},{}}]
	local output3 = output[{{batch_size*2+1,batch_size*3},{}}]
	local tmp_v1, tmp_index1 = torch.max(output2,2)
	local tmp_v2, tmp_index2 = torch.max(output3,2)

	for l=1,batch_size do
		output1[l][tmp_action1[l]] = tmp_v1[l] + gamma * tmp_reward1[l] 
		output1[l][tmp_action2[l]] = tmp_v2[l] + gamma * tmp_reward2[l] 
	end
	training_set.data = tmp_training_data[{{1,batch_size}}]
	training_set.label = output1

	return training_set
end




function func_construct_dqn_pvn_training_data( minibatch, training_set, training_set_pvn, dqn,gamma )
	local batch_size = #minibatch
	local tmp_training_data = torch.Tensor(batch_size*3, input_vector_size):cuda()
	local tmp_action1 = {}
	local tmp_action2 = {}
	local tmp_reward1 = torch.Tensor(batch_size, 1):fill(0):cuda()
	local tmp_reward2 = torch.Tensor(batch_size, 1):fill(0):cuda()
	local index_enable={}

	for l,memory in pairs(minibatch) do
		tmp_training_data[l] = memory[1]
		tmp_training_data[l+batch_size] = memory[6]
		tmp_training_data[l+2*batch_size] = memory[7]

		table.insert(tmp_action1, memory[2])
		table.insert(tmp_action2, memory[3])

		tmp_reward1[l] = memory[4]
		tmp_reward2[l] = memory[5]
		if memory[8] == 1 then
			training_set_pvn.label[l][1] = 1
			table.insert(index_enable,l)
		elseif memory[8] == -1 then
			training_set_pvn.label[l][1] = 2
			table.insert(index_enable,l)
		end
	end
	local output = dqn:forward(tmp_training_data)
	local output1 = output[{{1,batch_size}, {}}]:clone()
	local output2 = output[{{batch_size+1,2*batch_size},{}}]
	local output3 = output[{{batch_size*2+1,batch_size*3},{}}]
	local tmp_v1, tmp_index1 = torch.max(output2,2)
	local tmp_v2, tmp_index2 = torch.max(output3,2)
	
	for l=1,batch_size do
		output1[l][tmp_action1[l]] = tmp_v1[l] + gamma * tmp_reward1[l] 
		output1[l][tmp_action2[l]] = tmp_v2[l] + gamma * tmp_reward2[l] 
	end

	training_set.data = tmp_training_data[{{1,batch_size}}]
	training_set.label = output1

	training_set_pvn.data = training_set.data:index(1,torch.LongTensor(index_enable))
	training_set_pvn.data = training_set_pvn.data[{{},{1,feature_size}}]
	training_set_pvn.label = training_set_pvn.label:index(1,torch.LongTensor(index_enable))


	return training_set, training_set_pvn
end


--[[
function func_construct_dqn_pvn_training_data( minibatch, training_set, training_set_pvn, dqn,gamma )
	for l, memory in pairs(minibatch) do
		local tmp_input_vector = memory[1]
		local tmp_action1 = memory[2]
		local tmp_action2 = memory[3]
		local tmp_reward1 = memory[4]
		local tmp_reward2 = memory[5]
		local tmp_new_input_vector1 = memory[6]
		local tmp_new_input_vector2 = memory[7]
		local tmp_obj_y_o_n = memory[8]
		local old_action_output = dqn:forward(tmp_input_vector)
		local new_action_output1 = dqn:forward(tmp_new_input_vector1)
		local new_action_output2 = dqn:forward(tmp_new_input_vector2)
		local y = old_action_output:clone()
		local tmp_v, tmp_index = torch.max(new_action_output1,1)
		tmp_v = tmp_v[1]
		tmp_index = tmp_index[1]
		local update_reward = tmp_reward1 + gamma * tmp_v
		y[tmp_action1] = update_reward
		tmp_v, tmp_index = torch.max(new_action_output2,1)
		tmp_v = tmp_v[1]
		tmp_index = tmp_index[1]
		local update_reward = tmp_reward2 + gamma * tmp_v
		y[tmp_action2] = update_reward
		training_set.data[l] = tmp_input_vector
		training_set.label[l] = y
		training_set_pvn.data[l] = tmp_input_vector[{{1,feature_size}}]
		training_set_pvn.label[l] = tmp_obj_y_o_n
	end
	return training_set, training_set_pvn
end
--]]

function func_construct_rgn_training_data(resnet_buffer,training_set)
	for l,memory in pairs(resnet_buffer) do
		training_set.data[l] = memory[1]
		training_set.label[l] = memory[2]
	end
	return training_set
end

function func_construct_resnet_training_data(resnet_buffer,training_set)
	for l,memory in pairs(resnet_buffer) do
		training_set.data[l] = memory[1][1]
		training_set.label[l] = memory[2]
	end
	return training_set
end