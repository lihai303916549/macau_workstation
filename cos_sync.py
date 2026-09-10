"""腾讯云 COS 上传脚本（老六免梯子部署用）

本地单文件上传：
    python cos_sync.py <本地index.html路径>

本地整目录同步（排除指定文件）：
    python cos_sync.py --dir <目录> [排除逗号列表]
    python cos_sync.py --dir vercel_deploy update.js

GitHub Actions 自动上传（密钥从 secrets 环境变量读取，绝不硬编码）：
    COS_SECRET_ID / COS_SECRET_KEY / COS_BUCKET / COS_REGION 由 workflow 注入
"""
import os
import sys
from qcloud_cos import CosConfig, CosS3Client

DEFAULT_REGION = "ap-shanghai"
DEFAULT_BUCKET = "laoliu-1485521374"

CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def make_client(secret_id, secret_key, region=DEFAULT_REGION):
    return CosS3Client(CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key))


def upload_file(c, bucket, local_path, key, acl="public-read"):
    ext = os.path.splitext(local_path)[1].lower()
    ct = CTYPES.get(ext, "application/octet-stream")
    with open(local_path, "rb") as f:
        c.put_object(Bucket=bucket, Body=f, Key=key, ContentType=ct,
                     ACL=acl, ContentDisposition="inline")
    print(f"  ✅ {key}  ({ct}, inline)")


def upload(secret_id, secret_key, src, key="index.html",
           bucket=DEFAULT_BUCKET, region=DEFAULT_REGION):
    c = make_client(secret_id, secret_key, region)
    upload_file(c, bucket, src, key)


def sync_dir(secret_id, secret_key, local_dir, bucket=DEFAULT_BUCKET,
             region=DEFAULT_REGION, exclude=()):
    c = make_client(secret_id, secret_key, region)
    uploaded = 0
    for name in sorted(os.listdir(local_dir)):
        if name in exclude:
            continue
        lp = os.path.join(local_dir, name)
        if os.path.isfile(lp):
            upload_file(c, bucket, lp, name)
            uploaded += 1
    print(f"📦 已同步 {uploaded} 个文件到 {bucket} ({region})")


def main():
    secret_id = os.environ.get("COS_SECRET_ID")
    secret_key = os.environ.get("COS_SECRET_KEY")
    region = os.environ.get("COS_REGION", DEFAULT_REGION)
    bucket = os.environ.get("COS_BUCKET", DEFAULT_BUCKET)
    if not secret_id or not secret_key:
        print("⚠️ 未设置 COS_SECRET_ID / COS_SECRET_KEY，跳过 COS 同步（不影响网页自动更新）")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--dir":
        d = sys.argv[2]
        ex = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else ()
        sync_dir(secret_id, secret_key, d, bucket, region, ex)
    else:
        src = sys.argv[1] if len(sys.argv) > 1 else "vercel_deploy/index.html"
        key = os.environ.get("COS_KEY", "index.html")
        upload(secret_id, secret_key, src, key, bucket, region)


if __name__ == "__main__":
    main()
