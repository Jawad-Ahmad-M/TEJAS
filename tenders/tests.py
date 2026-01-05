'''def calculate_checksum(file):
    sha256_hash = hashlib.sha256()
    for chunk in file.chunks():
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()  # 64-char hex'''