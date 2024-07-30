from clifspipe.cube import generate_cube
import warnings
import toml

def run_clifs_pipeline(config):
    generate_cube(config)

if __name__ == "__main__":
    config_file = "/arc/projects/CLIFS/config_files/clifs_39.toml"
    config = toml.load(config_file)
    if config["pipeline"]["suppress_warnings"]:
        warnings.filterwarnings("ignore")
        
    run_clifs_pipeline(config)
    