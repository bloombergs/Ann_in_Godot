extends Node2D

onready var text_edit = $TextEdit
onready var submit_button = $Button
onready var result_label = $Label

var weights: float
var bias: float
var user_input: int = 0 
var final_loss : float = 0

func custom_relu(x: float) -> float:
	return max(0, x)

func custom_relu_derivative(z: float) -> int:
	return 1 if z > 0 else 0

func forward_propagation(x: float) -> Array:
	var z = x * weights + bias
	var a = custom_relu(z)
	return [a, z]

func train_model():
	var x_train = [1, 2, 3, 4, 5]
	var y_train = [2, 4, 6, 8, 10]
	var learning_rate = 0.0001
	var epochs = 2000

	for epoch in range(epochs):
		var predictions = []
		var intermediate_outputs = []
		for x in x_train:
			var result = forward_propagation(x)
			var pred = result[0]
			var inter_out = result[1]
			predictions.append(pred)
			intermediate_outputs.append(inter_out)
		
		var loss = 0.0
		for i in range(predictions.size()):
			loss += pow(predictions[i] - y_train[i], 2)
		loss /= predictions.size()

		var dL_dw = 0.0
		var dL_db = 0.0
		for i in range(predictions.size()):
			var dL_da = 2 * (predictions[i] - y_train[i]) / predictions.size()
			var da_dz = custom_relu_derivative(intermediate_outputs[i])
			dL_dw += x_train[i] * dL_da * da_dz
			dL_db += dL_da * da_dz
		
		weights -= learning_rate * dL_dw
		bias -= learning_rate * dL_db
		
		if epoch % 100 == 0:
			print("Epoch %d: Loss %f" % [epoch, loss])
		final_loss = loss

func _ready():
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	weights = rng.randf_range(-1, 1)
	bias = rng.randf_range(-1, 1)
	
	submit_button.connect("pressed", self, "_on_Button_pressed")

func _on_Button_pressed():
	var input_text = text_edit.text.strip_edges()
	if not input_text.is_valid_integer():
		result_label.text = "Please enter a valid integer."
		return
	
	user_input = int(input_text)
	
	train_model()
	
	var result = forward_propagation(user_input)
	var prediction = result[0]
	result_label.text = "Prediction for input " + str(user_input) + ": " + str(prediction) + "\n" + "final loss : " + str(final_loss)
