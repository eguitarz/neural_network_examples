require 'ruby-fann'

def is_acceptable?(fann)
	( fann.run([0,1]).first - 1 ).abs < 0.01 and
	( fann.run([0,0]).first - 0 ).abs < 0.01
end

begin
	train = RubyFann::TrainData.new(:inputs=>[[0, 0], [0, 1], [1, 0], [1, 1]], :desired_outputs=>[[0], [1], [1], [0]])
	fann = RubyFann::Standard.new(:num_inputs=>2, :hidden_neurons=>[3, 3], :num_outputs=>1)
	fann.train_on_data(train, 1000, 10, 0.1) # 1000 max_epochs, 10 errors between reports and 0.1 desired MSE (mean-squared-error)
	outputs = fann.run([0, 1]) 
	puts '### 1 ###'
	puts outputs
	train.save('verify.train')
	train = RubyFann::TrainData.new(:filename=>'verify.train')
	# Train again with 10000 max_epochs, 20 errors between reports and 0.01 desired MSE (mean-squared-error)
	# This will take longer:
	fann.train_on_data(train, 10000, 20, 0.01) 
end while !is_acceptable? fann
fann.save('foo.net')
saved_nn = RubyFann::Standard.new(:filename=>"foo.net")
puts '### 2 ###'
puts saved_nn.run([1, 1]) 