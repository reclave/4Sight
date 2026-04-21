import os
import yaml
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials

# Refer to Cognite Docs
def client_gen(env):
    config = read_config(env)
    client = CogniteClient(ClientConfig(
        credentials=OAuthClientCredentials(token_url=config["token_url"], client_id=config["client_id"],
                                           client_secret=config["client_secret"], scopes=config["token_scopes"]),
        project=config["cognite_project"],
        base_url=f"https://{config['cdf_cluster']}.cognitedata.com",
        client_name=config["client_name"],
        debug=False,
        timeout=90,
    ))
    return client

def read_config(env):
    # Adjust directory below to local secrets.yaml file (testing/dev)
    config_path = os.path.expanduser(f"~/.secrets/{env}-client.yaml")
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.SafeLoader)
    return config

if __name__ == '__main__':
    print(client_gen('dev').iam.token.inspect())