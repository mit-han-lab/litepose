from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(
    name='fast_utils',
    version='0.0.1',
    description='Pose detection accelerated by NVIDIA TensorRT',
    packages=find_packages(),
    ext_package='fast_utils',
    ext_modules=[cpp_extension.CppExtension('plugins', [
        'fast_utils/parse/find_peaks.cpp',
        'fast_utils/parse/assign.cpp',
        'fast_utils/plugins.cpp',
    ])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
    ],
)
