from argparse import ArgumentParser
import os

import requests
from tqdm import tqdm
from urllib.parse import urlencode


def download_file(public_key: str, fname: str):
    file_path = os.path.dirname(fname)
    if len(file_path) > 0:
        if not os.path.isdir(file_path):
            raise ValueError(f'The directory "{file_path}" does not exist!')
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    final_url = base_url + urlencode(dict(public_key=public_key))
    pk_request = requests.get(final_url)
    direct_link = pk_request.json().get('href')
    response = requests.get(direct_link, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(fname, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if (total_size_in_bytes != 0) and (progress_bar.n != total_size_in_bytes):
        raise ValueError(f'The file "{fname}" is not loaded!')


def main():
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', dest='file_name', type=str, required=True,
                        help='The saved file name.')
    parser.add_argument('-u', '--url', dest='yandex_disk_url', type=str, required=False, default=None,
                        help='The Yandex Disk URL (public key).')
    args = parser.parse_args()

    download_file(public_key=args.yandex_disk_url, fname=os.path.normpath(args.file_name))


if __name__ == "__main__":
    main()
