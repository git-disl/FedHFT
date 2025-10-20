class Config:
    def __init__(self, args):

        self.training_params = {
            'model_default': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 1,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                }
            },
            'bert-base-uncased': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'cola': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 3e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'sst2': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'mrpc': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'qqp': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'mnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'qnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'wnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-2,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'rte': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 3e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'ag_news': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                }
            },
            'bert-large-uncased': {
                'data_default': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'cola': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 3e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'sst2': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'mrpc': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'qqp': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'mnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'qnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-3,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'wnli': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-2,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'rte': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 3e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                },
                'ag_news': {
                    'init': {
                        'num_train_epochs': 2,
                        'learning_rate': 1e-4,
                        'weight_decay': 1e-2,
                        'batch_size': 32
                    }
                }
            }
        }
        self.adapter_params = {
            'model_default': {
                'data_default': {
                    'r': 32,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1,
                    'target_modules': ["query", "value"],
                    'task_type': 'SEQ_CLS'
                }
            }
        }
        self.quant_params = {
            'model_default': {
                'data_default': {
                }
            }
        }

    def get_init_training_params(self, model_name, data_name):
        default_params = self.training_params.get(model_name, self.training_params['model_default'])['data_default']['init']
        data_params = self.training_params.get(model_name, self.training_params['model_default']).get(data_name, {'init': {}})['init']
        return default_params | data_params

    def get_adapter_params(self, model_name, data_name):
        default_params = self.adapter_params.get(model_name, self.adapter_params['model_default'])['data_default']
        data_params = self.adapter_params.get(model_name, self.adapter_params['model_default']).get(data_name, {})
        return default_params | data_params

    def get_quant_params(self, model_name, data_name):
        default_params = self.quant_params.get(model_name, self.quant_params['model_default'])['data_default']
        data_params = self.quant_params.get(model_name, self.quant_params['model_default']).get(data_name, {})
        return default_params | data_params
