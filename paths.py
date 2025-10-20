def get_path(args, key: str, add_date=False, temp=True):

    method_tag = get_method_tag(args)

    if add_date:
        if method_tag:
            method_tag += '/'
        method_tag += args.datetime[:19]

    method_tag = method_tag.replace(' ', '_')

    folder_prefix = f'{args.save_path}/{args.task}/{args.data}/{args.arch}/{args.comp_method}/{method_tag}'

    if temp:
        folder_prefix += '/temp'

    trainer_prefix = f'{folder_prefix}/trainer'
    model_prefix = f'{folder_prefix}/models'
    figure_prefix = f'{folder_prefix}/figures'

    path_dict = {'MAIN_FOLDER_DIR': folder_prefix,
                 'TRAINER_FOLDER_DIR': trainer_prefix,
                 'MODEL_FOLDER_DIR': model_prefix,
                 'FIGURE_FOLDER_DIR': figure_prefix,
                 'TEST_PREDS_PATH': f'{model_prefix}/test_preds.pth',
                 'ARGS_PATH': f'{folder_prefix}/args.json'
                 }

    return path_dict[key]


def get_method_tag(args):
    method_tag = []
    if 'a' in args.comp_method:
        method_tag.append(args.adapter_method)
    if 'q' in args.comp_method:
        method_tag.append(args.quant_method)

    method_tag = '_'.join(method_tag)
    if not method_tag:
        method_tag = '_'

    return method_tag
