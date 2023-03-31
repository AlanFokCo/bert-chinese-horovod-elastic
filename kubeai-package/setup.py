from setuptools import setup, find_packages

setup(name='kubeai',
      version='0.0.2',
      description='Python SDK of Alibaba Cloud KubeAI',
      author='AlanFok',
      install_requires=["pymysql"],
      author_email='huozhixin.hzx@alibaba-inc.com',
      packages=find_packages()
)
