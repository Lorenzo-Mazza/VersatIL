import argparse
import json
from diffusion_endpoint import main as diffusion_main
from flow_matching_endpoint import main as flow_matching_main
from act_endpoint import main as act_main
from phase_act_endpoint import main as phase_act_main
from legacy_config import DiffusionConfig, FlowMatchingConfig, ACTConfig, PhaseACTConfig
from legacy_constants import PolicyType

def override_default_config(config: DiffusionConfig, params_to_override: dict):

    for key, value in params_to_override.items():
        if key != "policy_name":
            if key in config.__dict__:
                setattr(config, key, value)
            else:
                raise ValueError(f"Key {key} not found in config")

    return config

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--custom_config_path", type=str, required=True)
    args = parser.parse_args()
    params = json.load(open(args.custom_config_path))
    policy_name = params["policy_name"]

    if policy_name == PolicyType.DIFFUSION_POLICY.value:
        config = DiffusionConfig()
        config = override_default_config(config, params)
        diffusion_main(config)
    elif policy_name == PolicyType.FLOW_MATCHING.value:
        config = FlowMatchingConfig()
        config = override_default_config(config, params)
        flow_matching_main(config)
    elif policy_name == PolicyType.ACT.value:
        config = ACTConfig()
        config = override_default_config(config, params)
        act_main(config)
    elif policy_name == PolicyType.PHASE_ACT.value:
        config = PhaseACTConfig()
        config = override_default_config(config, params)
        phase_act_main(config)
    else:
        raise ValueError(f"Policy name {policy_name} not found")

if __name__ == "__main__":
    main()