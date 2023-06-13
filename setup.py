import setuptools

from src import __version__


setuptools.setup(
    name="pytorch-incomplete_classification",
    version=__version__,
    author="Junfei Ren",
    author_email="20215227018@stu.suda.edu.cn",
    description="incomplete_classification",
    packages=setuptools.find_packages(
        exclude=["data"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "omegaconf>=2.0.6",
        "tqdm>=4.61.1",
        "pytorch-rex==0.0.15",
        "lxml>=4.6.3",
        "transformers>=4.12.5",
    ],
)
