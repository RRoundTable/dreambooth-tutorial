from diffusers import StableDiffusionPipeline
import torch
import argparse

def load_args():
    parser = argparse.ArgumentParser(prog='Stable Diffusion Inference')
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--prompt", type=str, default="A photo of sks dog in a bucket")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--output_dir", type=str, default="save")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_args()
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32).to("cuda")

    image = pipe(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale
    ).images[0]
    image.save(f"save/{args.model_id}-{args.prompt}.png")