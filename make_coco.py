from PIL import Image
import torch
from tqdm.auto import tqdm
import argparse
import json

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from rembg import remove as rm

ANNOTATION_NUM = 591752


def get_bit(num: int) -> int:
    if num == 0:
        return 1
    assert num > 0
    cnt = 0
    while num != 0:
        cnt += 1
        num = num // 10
    return cnt

def id2name(id: int) -> str:
    cc = 12
    cn = get_bit(id)
    return '0' * (cc - cn) + str(id)


def get_base_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='dir to write results to'
    )
    parser.add_argument(
        '--coco_path',
        type=str,
        default=None,
        help='direction to read COCO2017 Dataset '
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

    return parser


def main():
    parser = get_base_argument_parser()
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    print('creating base model...')
    base_name = opt.base_model  # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('getting base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('getting upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[opt.min_points, opt.max_points],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )



    assert not opt.input_txt.endswith('.txt'), 'Invalid Input of input txt path'
    opt.input_txt = opt.input_txt if opt.input_txt.endswith('/') else opt.input_txt + '/'
    opt.outdir = opt.outdir if opt.outdir.endswith('/') else opt.outdir + '/'
    opt.coco_path = opt.coco_path if opt.coco_path.endswith('/') else opt.coco_path + '/'
    assert opt.coco_path.endswith('coco2017/'), 'coco dataset path ends with \'coco2017\' needed'

    ori_img = opt.outdir + 'ori_img/'    # rewrite and rename coco images into the new folder
    point_img = opt.outdir + 'point_img/'
    prompt_path = opt.outdir + 'prompts/'

    prompt_file = open(prompt_path + 'prompts.txt', 'a')

    json_path = opt.coco_path + 'annotations/captions_train2017.json'
    image_path = opt.coco_path + 'train2017/'
    json_caption = open(json_path)
    json_file = json.load(json_caption)
    annotation = json_file['annotation']
    for num in range(ANNOTATION_NUM):
        NUM = str(num)
        image_id = (int)(annotation[NUM]['image_id'])
        caption = annotation[NUM]['caption']
        image_id = id2name(image_id)
        image_name = image_id + '.jpg'
        img = Image.open(image_path + image_name)
        img = rm(img)
        img.save(ori_img + image_name)              # original image
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=1, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
        fig.savefig(point_img + image_name)         # image removed background
        prompt_file.write(caption + '\n')



if __name__ == "__main__":
    main()




