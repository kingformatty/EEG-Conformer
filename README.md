
Dataloader:

	Please modify path in "src/dataset_eeg.py" to match your implementation. 
		
		
Train Mode:

		Open "training.py", specify hyperparameters in the declaration, the modifiable hyperparameters are as follows:
			- BATCH_SIZE
			- NUM_WORKERS
			- LR (learning rate)
			- weidht_decay (l2 regularization strength)
			- EPOCHS
			- d_model (encoder dimension in the work flow)
			- q (query size for attention)
			- v (value size for attention)
			- h (number of heads for multi-head attention)
			- N (number of Conformer blocks to stack)
			- attention_size (attention window size)
			- dropout
			- pe (position encoding option, "regular"/ "original" / None
			- only encoder (only use encoder or encoder-decoder, True/False)
			- d_input (input feature dimension, fixed to be 22)
			- d_output (output logits dimension, fixed to be 4)
			- embedding_option (embedding output option, "mean"/"last")
			- crop (sub_sampling size)
			- conformer (whether use conformer or base transformer, True/False)
			- conv_dim (1D conv or 2D conv, "feature"/"channel")
			- normalize (whether to normalize the data or not, True/False)

Eval Mode:

		Open "load_model.py", specify the path to the best model.
		For examples:
			path =  '/media/kingformatty/easystore/C247/project/transformer/exp/general crop/conformer_lr_dc0.01_4conv_kernel_channel_last_crop800_normalize/models/best_model26.pth'
		The script will not read the configuration automatically, please specify the model configuration in the configuration part manually. 
		Test accuracy will be printed out in terminal.
		
		For your convenience, we provide the best model in "model" folder. And default configuration is set to match this configuration. By running the script, the test accuracy from my side is 61.625%. Please modify the model path to match your implementation.
