import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inno-graph",
    version="1.0",
    author="cutefluffyfox",
    author_email="several authors",
    description="Python library for GCN tutorial",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cutefluffyfox/yalice",
    packages=setuptools.find_packages(),
    install_requires=['numpy~=1.22.0', 'setuptools~=59.6.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
