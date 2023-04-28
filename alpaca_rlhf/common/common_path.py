import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# project_dir = '/my-alpaca/MyDrive/my-alpaca'

templates_dir = os.path.join(project_dir, 'templates')

data_dir = os.path.join(project_dir, 'data/')

if __name__ == '__main__':
    print(project_dir)
