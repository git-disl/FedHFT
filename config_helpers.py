from peft import LoraConfig, TaskType


def get_adapter_config(adapter_params_dict):
    r = adapter_params_dict['r']
    lora_alpha = adapter_params_dict['lora_alpha']
    lora_dropout = adapter_params_dict['lora_dropout']
    task_type = TaskType.__getattr__(adapter_params_dict["task_type"])

    peft_config = LoraConfig(
        task_type=task_type, inference_mode=False, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=adapter_params_dict['target_modules']
    )

    return peft_config
