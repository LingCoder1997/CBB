from setuptools import setup, find_packages

setup(
    name="CBB",
    version="1.0",  # 版本号
    author="Lingxuan_Wang",  # 作者
    author_email="lingxuanwang123@163.com",  # 作者邮箱
    description="This is a simple version of the common block lib ",  # 包描述
    url="https://github.com/your_username/myhello",  # 项目地址
    packages=find_packages(),  # 包含所有包
    classifiers=[  # 分类器列表
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # 要求的Python版本
)
