require 'torch'

function func_read_training_cmd(cmd, arg)
	cmd:text()
	cmd:text('Training Agent:')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-dataset', 2, 'Training dataset, 1 for 2012, 2 for 2007&2012')
	cmd:option('-load_saved_data', 1, 'if data saved, don\'t need to parse the annotation files again' )
	cmd:option('-load_saved_testdata', 0,'test data')
	cmd:option('-finetune_resnet', 0, 'if finetune resnet while training dqn')
	cmd:option('-lr', 0.01, 'learning rate')
	cmd:option('-lrd',0.00009, 'learning rate decay')
	cmd:option('-replay_buffer', 50000, 'replay memory buffer size')
	cmd:option('-enable_evaluate', 0, 'enable validate after each epoch')
	--cmd:option('-alpha', 0.2, 'action scalar, default')
	cmd:option('-dqn_log', './log/v_log', 'dqn log file')
	cmd:option('-rgn_log', './log/r_log', 'rgn log file')
	cmd:option('-pvn_log', './log/p_log', 'pvn log file')
	cmd:option('-resnet_log', './log/n_log', 'resnet log file')
	cmd:option('-max_steps', 20, 'max step for one clip, default ')
	cmd:option('-batch_size', 500, 'batch size, default')
	cmd:option('-epochs', 25, 'epochs, default')
	cmd:option('-enable_CAM', 0, '1 to enable CAM; default 0')
	cmd:option('-enable_background', 0, 'model tag')
	cmd:option('-enable_PVN', 0,'')
	cmd:option('-enable_RGN', 0, '')
	cmd:option('-enable_glb_view', 0, '')
	cmd:option('-enable_history_action', 1 ,'')
	cmd:option('-enable_loc_map', 0,'')
	cmd:option('-delay_update', 0,'')
	cmd:option('-name', 'a', 'model tag')
	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt
end


function func_read_test_cmd(cmd, arg)
	cmd:text()
	cmd:text('Test Agent:')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-dataset', 1, 'Training dataset, 1 for 2007, 2 for 2012')
	cmd:option('-load_saved_testdata', 0,'test data')
	cmd:option('-max_steps', 31, 'max step for one clip, default ')
	cmd:option('-enable_CAM', 0, '1 to enable CAM; default 0')
	cmd:option('-enable_rgn', 1, '1/0')
	cmd:option('-name', 'a', 'model tag')
	cmd:option('-resnet','','resnet model')
	cmd:option('-dqn','','dqn model')
	cmd:option('-rgn','','rgn model')
	cmd:option('-pvn','','pvn model')
	cmd:option('-topk', 1, 'top k results used to calculate CAM')
	cmd:option('-enable_glb_view', 0, '')
	cmd:option('-enable_loc_map', 0,'')
	cmd:option('-enable_history_action', 1 ,'')
	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt
end

function func_read_visualization_cmd(cmd, arg)
	cmd:text()
	cmd:text('Test Agent:')
	cmd:text()
	cmd:text('Options:')
	

	cmd:option('-img', 'a', 'model tag')
	cmd:option('-resnet','','resnet model')
	cmd:option('-dqn','','dqn model')

	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt

end