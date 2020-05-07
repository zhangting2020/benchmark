import argparse
import random
import os
import json

def select_configs(args, logs):
    config_groups = grouping_configs(logs, args.ignored_params)
    print("==================config_groups===================")
    max_num_configs = args.max_num_configs
    if not max_num_configs:
        max_num_configs = len(config_groups)
    selected_config_ids = []
    for key in config_groups:
        print("config: {0}, total: {1}".format(key, len(config_groups[key])))
        config_ids = config_groups[key]
        number = max(1, max_num_configs*len(config_ids)/len(logs))
        ids = random.sample(config_ids, int(number))
        print("Select {0} config_ids: {1}.".format(number, ids))
        selected_config_ids += ids

    with open(args.json_file, 'r') as f:
        all_configs = json.load(f)
    out_dir = os.path.dirname(args.output_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configs = []
    for index in selected_config_ids:
        configs.append(all_configs[index])
    with open(args.output_file, 'w') as f:
        json.dump(configs, f, indent=4, sort_keys=True)


def grouping_configs(logs, ignored_params=None):
    config_groups = dict()
    for i in range(len(logs)):
        config_str = remove_ignored_params(logs[i], ignored_params)
        if config_str not in config_groups.keys():
            config_groups[config_str] = [i]
        else:
            config_groups[config_str] += [i]
    return config_groups


def remove_ignored_params(log, ignored_params):
    log_src = list(log)
    if ignored_params:
        for item in log_src:
            param_val = item.split("=")
            if param_val[0] in ignored_params:
                log.remove(item)

    log_str = ' '.join(log)
    return log_str
    

def get_logs(op_name, log_file):
    logs = []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            keyword = "op=" + op_name
            if keyword in line:
                index = line.index(keyword)
                line = line[index:].replace(', ', ',').split(' ')
                logs.append(line)
    return logs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--op_name',
        type=str,
        default=None,
        help='Specify the operator name.')
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Specify the path of log file.')
    parser.add_argument(
        '--json_file',
        type=str,
        default=None,
        help='Specify the path of json file.')
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Specify the path of json file.')
    parser.add_argument(
        '--ignored_params',
        nargs='*',
        help='Specify the ignored param list, the configs will be filtered according to the other params.')
    parser.add_argument(
        '--max_num_configs',
        type=int,
        default=None,
        help='Specify the maximum number of selected configs.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("ignored_params: {0}.".format(args.ignored_params))
    logs = get_logs(args.op_name, args.log_file)
    select_configs(args, logs)

