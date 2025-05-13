import argparse
from ui.components import create_main_demo_ui
from pipeline_ace_step import ACEStepPipeline
from data_sampler import DataSampler
import os


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--server_name", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--share", action='store_true', default=False)
parser.add_argument("--bf16", action='store_true', default=True)
parser.add_argument("--torch_compile", type=bool, default=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)


persistent_storage_path = "/data"


def main(args):

    model_demo = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        persistent_storage_path=persistent_storage_path,
        torch_compile=args.torch_compile
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
        load_data_func=data_sampler.load_json,
    )
    demo.queue(default_concurrency_limit=8).launch(
        
    )


if __name__ == "__main__":
    main(args)
