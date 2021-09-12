from setuptools import find_packages, setup

# Load README file
with open(file="README.md", mode="r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name="elizastarr_caption_generator",
    packages=find_packages(),
    version="0.1.0",
    description="Code for the second technical interview at PTTRNS.ai.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "matplotlib", "tensorflow"],
    author="Eliza Starr",
    author_email="eliza.r.starr@gmail.com",
    license="MIT",
)
