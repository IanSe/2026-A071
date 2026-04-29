import os
from pathlib import Path
from string import Template
import yaml
import shutil


def dockerfile_template():
    """
    This function creates the Dockerfile for the model.
    It also creates the model directory.
    It also copies the template.Dockerfile file to the model directory.
    """
    dockerfile_template_path = "./template.Dockerfile"
    parameters_path = "./params.yml"

    params = yaml.safe_load(Path(parameters_path).read_text())
    dockerfile_template = Template(Path(dockerfile_template_path).read_text())

    dockerfile_file = dockerfile_template.substitute(
        MODEL_NAME=params['MODEL_NAME'],
        SIZE=params['SIZE'],
        PYTORCH_VERSION=params['PYTORCH_VERSION'],
    )

    os.makedirs(f"./{params['MODEL_NAME']}/{params['SIZE']}", exist_ok=True)

    with open(f"./{params['MODEL_NAME']}/{params['SIZE']}/{params['MODEL_NAME']}-{params['SIZE']}.Dockerfile", "w") as f:
        f.write(dockerfile_file)

    print(f"Dockerfile created for {params['MODEL_NAME']} {params['SIZE']} at ./{params['MODEL_NAME']}/{params['SIZE']}/Dockerfile")



def compose_template():
    """
    This function creates the docker-compose.yml file for the model.
    It also creates the grafana and metrics directories.
    It also copies the prometheus.yml file to the metrics directory.
    """

    compose_template_path = "./docker-compose-template.yml"
    parameters_path = "./params.yml"

    params = yaml.safe_load(Path(parameters_path).read_text())
    compose_template = Template(Path(compose_template_path).read_text())

    compose_file = compose_template.substitute(
        MODEL_NAME=params['MODEL_NAME'],
        SIZE=params['SIZE'],
        PYTORCH_VERSION=params['PYTORCH_VERSION'],
        PROMETHEUS_VERSION=params['PROMETHEUS_VERSION'],
        GRAFANA_VERSION=params['GRAFANA_VERSION'],
        NODE_EXPORTER_VERSION=params['NODE_EXPORTER_VERSION'],
        AMD_DME_VERSION=params['AMD_DME_VERSION'],
        KEPLER_VERSION=params['KEPLER_VERSION'],
    )

    with open(f"./{params['MODEL_NAME']}/{params['SIZE']}/docker-compose.yml", "w") as f:
        f.write(compose_file)

    print(f"Compose file created for {params['MODEL_NAME']} {params['SIZE']} at ./{params['MODEL_NAME']}/{params['SIZE']}/docker-compose.yml")

    os.makedirs(f"./{params['MODEL_NAME']}/{params['SIZE']}/grafana", exist_ok=True)
    os.makedirs(f"./{params['MODEL_NAME']}/{params['SIZE']}/metrics", exist_ok=True)

    prometheus_template = Template(Path("./prometheus.yml").read_text())
    prometheus_file = prometheus_template.substitute(
        MODEL_NAME=params['MODEL_NAME'],
    )
    with open(f"./{params['MODEL_NAME']}/{params['SIZE']}/prometheus.yml", "w") as f:
        f.write(prometheus_file)

    shutil.copy('./.env', f"./{params['MODEL_NAME']}/{params['SIZE']}/.env")

    python_file_path = f"./{params['MODEL_NAME']}/{params['SIZE']}/{params['MODEL_NAME']}-{params['SIZE']}.py"
    with open(python_file_path, "w") as f:
        f.close()


def main():
    dockerfile_template()
    compose_template()


if __name__ == "__main__":
    main()
