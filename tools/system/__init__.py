import os

def get_info_sota():
    prw_root_dir = "/home/linhuadong/DNFNet/jobs/prw_sota"
    cuhk_root_dir = "/home/linhuadong/DNFNet/jobs/cuhk_sota"
    info = {
        'PRW':{
            "root_dir": prw_root_dir,
            "config": os.path.join(prw_root_dir, "work_dirs/prw/prw.py"),
            "checkpoint": os.path.join(prw_root_dir, "work_dirs/prw/latest.pth"),
        },
        'CUHK':{
            "root_dir": cuhk_root_dir,
            "config": os.path.join(cuhk_root_dir, "work_dirs/cuhk/cuhk.py"),
            "checkpoint": os.path.join(cuhk_root_dir, "work_dirs/cuhk/latest.pth"),
        }
    }
    return info

def get_info_baseline():
    prw_root_dir = "/home/linhuadong/DNFNet/jobs/prw_base"
    cuhk_root_dir = "/home/linhuadong/DNFNet/jobs/cuhk_sota"
    info = {
        'PRW':{
            "root_dir": prw_root_dir,
            "config": os.path.join(prw_root_dir, "work_dirs/prw/prw.py"),
            "checkpoint": os.path.join(prw_root_dir, "work_dirs/prw/latest.pth"),
        },
        'CUHK':{
            "root_dir": cuhk_root_dir,
            "config": os.path.join(cuhk_root_dir, "work_dirs/cuhk/cuhk.py"),
            "checkpoint": os.path.join(cuhk_root_dir, "work_dirs/cuhk/latest.pth"),
        }
    }

    return info