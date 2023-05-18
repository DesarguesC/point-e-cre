from setuptools import setup

setup(
    name="point-e",
    packages=[
        "point_e",
        "point_e.diffusion",
        "point_e.evals",
        "point_e.models",
        "point_e.util",
    ],
    install_requires=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "clip @ git+https://github.com/openai/CLIP.git",
        "rembg",
        "diffusers==0.12.1",
        "transformers==4.27.4",
        "accelerate==0.17.1"
    ],
    author="DesarguesC",
)
