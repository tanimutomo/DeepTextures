from glob import glob
import sys


if __name__ == '__main__':
    print(len(glob('data/{}/val/*/*.JPEG'.format(sys.argv[1]))))
