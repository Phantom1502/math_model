import os
import torch

def save_checkpoint(
    path        : str,
    model       : torch.nn.Module,
    optimizer   : torch.optim.Optimizer = None,
    scheduler                            = None,
    global_step  : int = 0,
    chunk_idx     : int = 0,
    val_loss       : float = None,
    extra           : dict = None,
    model_cfg        = None,
):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    ckpt = {
        "model"      : model.state_dict(),
        "global_step": global_step,
        "chunk_idx"  : chunk_idx,
        "val_loss"   : val_loss,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra:
        ckpt["extra"] = extra
    if model_cfg is not None:
        from dataclasses import asdict
        ckpt["model_cfg"] = asdict(model_cfg)

    torch.save(ckpt, path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(
    path        : str,
    model       : torch.nn.Module,
    optimizer   : torch.optim.Optimizer = None,
    scheduler                            = None,
    device                                = "cpu",
) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"  ✓ Loaded checkpoint ← {path}")
    print(f"    global_step={ckpt.get('global_step', 0)} | "
          f"chunk_idx={ckpt.get('chunk_idx', 0)} | "
          f"val_loss={ckpt.get('val_loss')}")

    return {
        "global_step": ckpt.get("global_step", 0),
        "chunk_idx"  : ckpt.get("chunk_idx", 0),
        "val_loss"   : ckpt.get("val_loss"),
        "extra"      : ckpt.get("extra"),
        "model_cfg"  : ckpt.get("model_cfg"),
        "file_order" : ckpt.get("file_order"),   # NEW — None nếu checkpoint cũ
    }
    
def hf_upload_latest(
    local_path : str,
    repo_id    : str,
    token      : str = None,
    filename   : str = "last_chunk.pt",
) -> bool:
    """
    Upload checkpoint lên HuggingFace Hub, lưu với tên cố định `filename`
    (mặc định "last_chunk.pt") — overwrite mỗi chunk để tiết kiệm storage.

    Args:
        local_path : đường dẫn file .pt local cần upload (thường là chunk_{idx}.pt)
        repo_id    : "username/repo-name" trên HuggingFace
        token      : HF token. Nếu None, đọc từ env HF_TOKEN hoặc từ
                     huggingface-cli login cache (~/.cache/huggingface/token).
        filename   : tên file trên HF repo (mặc định "last_chunk.pt")

    Returns:
        True nếu upload thành công, False nếu có lỗi (in cảnh báo, không raise).

    Yêu cầu:
        pip install huggingface_hub

    Setup (1 trong 3 cách):
        1. huggingface-cli login          (lưu token vào cache, tiện cho dev)
        2. export HF_TOKEN=hf_xxx...      (an toàn hơn cho Colab)
        3. truyền token=... trực tiếp     (tường minh nhất, không khuyến nghị hardcode)

    Repo phải tồn tại sẵn trên HuggingFace (type="model" hoặc "dataset").
    Tạo repo trước bằng:
        huggingface-cli repo create memlm-checkpoints --type model
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  ⚠ hf_upload_latest: huggingface_hub chưa cài. "
              "Chạy: pip install huggingface_hub")
        return False

    # Ưu tiên: tham số > env var > cache từ huggingface-cli login
    _token = token or os.environ.get("HF_TOKEN")

    try:
        api = HfApi(token=_token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"update {filename} ← {os.path.basename(local_path)}",
        )
        print(f"  ✓ Uploaded → hf:///{repo_id}/{filename}")
        return True

    except Exception as e:
        print(f"  ⚠ hf_upload_latest failed (training tiếp tục bình thường): {e}")
        return False
    
def hf_download_latest(
    repo_id     : str,
    local_path  : str,
    token       : str = None,
    filename    : str = "last_chunk.pt",
) -> bool:
    """
    Download checkpoint từ HuggingFace Hub về máy local (mặc định lấy file "last_chunk.pt").
    Ghi đè file local nếu đã tồn tại để tiết kiệm không gian lưu trữ.

    Args:
        repo_id    : "username/repo-name" trên HuggingFace
        local_path : đường dẫn local muốn lưu file (ví dụ: "./checkpoints/resume_chunk.pt")
        token      : HF token. Nếu None, đọc từ env HF_TOKEN hoặc cache của hệ thống.
        filename   : tên file cần kéo trên HF repo (mặc định "last_chunk.pt")

    Returns:
        True nếu download thành công, False nếu có lỗi (in cảnh báo, không raise).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ⚠ hf_download_latest: huggingface_hub chưa cài. "
              "Chạy: pip install huggingface_hub")
        return False

    # Ưu tiên: tham số > env var > cache từ huggingface-cli login
    _token = token or os.environ.get("HF_TOKEN")

    # Đảm bảo thư mục chứa file local tồn tại sẵn
    local_dir = os.path.dirname(local_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    try:
        print(f"  ⬇ Đang kéo file hf:///{repo_id}/{filename}...")
        
        # Tải file từ HF Hub thẳng về đường dẫn local mong muốn
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=_token,
            local_dir=local_dir if local_dir else ".",
            local_dir_use_symlinks=False, # Tránh dùng symlink để file vật lý nằm đúng chỗ
        )
        
        # Đổi tên file từ cache cấu trúc của HF sang chính xác local_path mong muốn (nếu cần)
        downloaded_path = os.path.join(local_dir if local_dir else ".", filename)
        if downloaded_path != local_path:
            if os.path.exists(local_path):
                os.remove(local_path) # Xóa file cũ trước khi rename để tránh xung đột trên Windows
            os.rename(downloaded_path, local_path)

        print(f"  ✓ Download thành công! File đã được lưu tại: {local_path}")
        return True

    except Exception as e:
        print(f"  ⚠ hf_download_latest failed (tiến trình tiếp tục bình thường): {e}")
        return False