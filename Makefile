MODEL_NAME=CompVis/stable-diffusion-v1-4
OUTPUT_DIR=dreambooth
LORA_OUTPUT_DIR=dreambooth-lora
INSTANCE_DIR=dog

env:
	conda create -n dreambooth python=3.9
setup:
	git clone https://github.com/huggingface/diffusers
	cd diffusers && pip install -e .
	cd diffusers/examples/dreambooth/ && pip install -r requirements.txt

train:
	# Train With use_8bit_adam, xformer for efficient training(GPU Memory Usage 9832MiB)
	accelerate launch train_dreambooth.py \
	--pretrained_model_name_or_path=$(MODEL_NAME)  \
	--instance_data_dir=$(INSTANCE_DIR) \
	--output_dir=$(OUTPUT_DIR) \
	--instance_prompt="a photo of sks dog" \
	--resolution=512 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=1 --gradient_checkpointing \
	--use_8bit_adam \
	--enable_xformers_memory_efficient_attention \
	--set_grads_to_none \
	--learning_rate=2e-6 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--num_class_images=200 \
	--max_train_steps=800

lora-train:
	accelerate launch train_dreambooth_lora.py \
	--pretrained_model_name_or_path=$(MODEL_NAME)  \
	--instance_data_dir=$(INSTANCE_DIR) \
	--output_dir=$(LORA_OUTPUT_DIR) \
	--instance_prompt="a photo of sks dog" \
	--resolution=512 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=1 \
	--checkpointing_steps=100 \
	--learning_rate=1e-4 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--max_train_steps=500 \
	--validation_prompt="A photo of sks dog in a bucket" \
	--validation_epochs=50 \
	--seed="0"

download:
	python download_dog.py