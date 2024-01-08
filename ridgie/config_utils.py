import yaml


def get_config() -> tuple:
    with open("conf.yaml") as conf_file:
        try:
            config = yaml.safe_load(conf_file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    n_dataset = config["n_dataset"]
    n_dimension = config["n_dimension"]
    weight_ones = config["weight_ones"]
    alpha = config["alpha"]

    return n_dataset, n_dimension, weight_ones, alpha
