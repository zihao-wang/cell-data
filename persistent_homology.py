from utils.data import Case3D
import ripserplusplus as rpp_py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="data/real/2022-06-17_D15_Pos0/stack_ROI.3d")

if __name__ == "__main__":
    case = Case3D(parser.parse_args().filename)
    rpp_py.run(
        "--format point-cloud",
        case.get_feature_label()[0]
    )
