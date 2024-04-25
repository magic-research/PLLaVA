import os.path as osp
import os
import re
import multiprocessing
import functools
import huggingface_hub
from huggingface_hub import snapshot_download


def upload(repo_id, local_dir, path_in_repo, repo_type, token):
    huggingface_hub.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        token=token,
        repo_type=repo_type
    )

def download(repo_id, local_dir, repo_type, token, filter_re=None):
    files = huggingface_hub.list_repo_files(repo_id, repo_type=repo_type, token=token)
    if filter_re is not None:
        files = [file for file in files if re.search(filter_re, file) is not None]
    pool = multiprocessing.Pool(8)
    download_func = functools.partial(
        huggingface_hub.hf_hub_download,
        repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=True,
        token=token
    )
    pool.map(download_func, files)
    print(f'downloaded files {files}')


def upload_file(repo_id, file_path, repo_type, token):
    huggingface_hub.upload_file(
        repo_id=repo_id,
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        token=token,
        repo_type=repo_type,
    )

if __name__ == '__main__':
    read_token = '...'
    write_token = '...'
    repo_id = '...'
    local_dir = '...'
    repo_type = '...'
    
    
    # #############
    # # Examples on most simple hf usage
    # # downlaod
    # filters = []
    # for filter_re in filters:
    #     download(repo_id,
    #              local_dir,
    #              repo_type,
    #              filter_re)

    # # upload
    # upload(repo_id, local_dir, local_dir, repo_type, write_token)
    # #############

    # download models
    repo_ids = [
        'ermu2001/plava-7b',
        'ermu2001/plava-13b',
    ]
    for repo_id in repo_ids:
        local_dir = repo_id.replace('ermu2001', 'MODELS')
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            repo_type='model',
            local_dir_use_symlinks=True,
            token=read_token,
        )