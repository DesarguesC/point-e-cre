from PIL import Image
import torch
from tqdm.auto import tqdm
import argparse

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from rembg import remove as rm


def get_base_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir_img',
        type=str,
        default=None,
        help='dir to write results to'
    )
    parser.add_argument(
        '--input_txt',
        type=str,
        default=None,
        help='direction to read prompts'
    )
    # prompt.txt, words.txt

    parser.add_argument(
        '--max_points',
        type=int,
        default=4096,
        help='max num of points in points cloud sampling'
    )
    parser.add_argument(
        '--min_points',
        type=int,
        default=1024,
        help='min num of points in points cloud sampling '
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='base40M',
        choices=['base40M', 'base300M', 'base1B'],
        help='which model to be used in base diffusion model'
    )

    parser.add_argument(
        '--bit_num',
        type=int,
        default=5,
        choices=[4,5,6,7,8,9],
        help='number amount'
    )


    return parser

def use_sd(prompt: str, save_dir: str,  name: str, model_id="stabilityai/stable-diffusion-2-1", remove=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = save_dir if save_dir.endswith('/') else save_dir + '/'
    name = name if name.endswith('.png') else name + '.png'

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    if remove:
        image = rm(image)
    image.save(save_dir + 'ori_image/' + name)
    return image, remove

def main():
    parser = get_base_argument_parser()
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert not opt.input_txt.endswith('.txt'), 'Invalid Input of input txt path'
    opt.input_txt = opt.input_txt if opt.input_txt.endswith('/') else opt.input_txt + '/'
    opt.outdir_img = opt.outdir_img if opt.outdir_img.endswith('/') else opt.outdir_img + '/'

    # create model
    print('creating base model...')
    base_name = opt.base_model # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[opt.min_points, opt.max_points],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    cnt_base = 0
    def get_bit(num: int) -> int:
        assert num >= 0
        cnt = 1
        while num != 0:
            cnt += 1
            num /= 10
        return cnt

    def get_name(opt, cnt_base: int) -> str:
        cc = opt.bit_num
        cn = get_bit(cnt_base)
        return '0'*(cc-cn) + str(cnt_base) + '.png'

    # Load an image to condition on.


    prompts = open(opt.input_txt + 'prompt.txt')
    lines = prompts.readlines()
    for line in lines:
        name = get_name(opt, cnt_base)
        cnt_base += 1
        img, _ = use_sd(line, opt.outdir_img, name, remove=True)
        # background of img has been removed

        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=1, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)),
                                remove_grid=True, single_side=True)
        fig.savefig(opt.outdir_img + 'point/' + name)



if __name__ == "__main__":
    main()




