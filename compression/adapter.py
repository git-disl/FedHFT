from config_helpers import get_adapter_config
from peft import get_peft_model, PeftModel


def inject_adapter(model, args, config, adapter_name='default'):
    peft_config = get_adapter_config(config.get_adapter_params(args.arch, args.data))
    if isinstance(model, PeftModel):
        model.add_adapter(adapter_name, peft_config)
    else:
        model = get_peft_model(model, peft_config, adapter_name)
    return model