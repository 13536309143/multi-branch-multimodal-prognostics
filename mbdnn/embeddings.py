"""Download only the original 100d ZIP member through HTTP Range requests."""
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from .data import read_rows, tokenize
from .utils import sha256, write_json

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.block_size = url, 0, 4 * 1024 * 1024
        self.cache_start, self.cache = -1, b""
        data, headers = self.request("bytes=-65536")
        self.size = int(headers["Content-Range"].split("/")[-1])
        self.etag = headers.get("ETag")
        self.cache_start, self.cache = self.size - len(data), data
        self.downloaded = len(data)

    def request(self, byte_range):
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, headers={"Range": byte_range, "User-Agent": "mbdnn-reproduction/0.1"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    if response.status != 206 or not response.headers.get("Content-Range"):
                        raise OSError("Server does not support Range; supply a local glove.6B.100d.txt instead")
                    headers = dict(response.headers)
                    if hasattr(self, "etag") and self.etag != headers.get("ETag"):
                        raise OSError("GloVe archive changed during download")
                    data = response.read()
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers["Content-Range"])
                    if not match or len(data) != int(match[2]) - int(match[1]) + 1:
                        raise OSError("Incomplete HTTP range")
                    if not byte_range.startswith("bytes=-") and int(match[1]) != int(byte_range[6:].split("-")[0]):
                        raise OSError("Server returned wrong offset")
                    return data, headers
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        output = bytearray()
        while remaining:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                start = self.position
                end = min(self.size - 1, start + max(self.block_size, remaining) - 1)
                self.cache, _ = self.request(f"bytes={start}-{end}")
                self.cache_start = start
                self.downloaded += len(self.cache)
            offset = self.position - self.cache_start
            length = min(remaining, len(self.cache) - offset)
            output.extend(self.cache[offset:offset + length])
            self.position += length
            remaining -= length
        return bytes(output)


def fetch_embeddings(root, destination, local_file=None):
    words = {word for row in read_rows(root) for word in tokenize(row["text"])}
    vectors = {}
    source = {"source": "glove.6B.100d", "dimension": 100, "url": GLOVE_URL}

    def consume(stream):
        for i, line in enumerate(stream):
            fields = line.split()
            if fields and fields[0] in words:
                if len(fields) != 101:
                    raise ValueError("Wrong GloVe dimension")
                vectors[fields[0]] = [float(v) for v in fields[1:]]
            if (i + 1) % 50000 == 0:
                print(f"GloVe lookup: {i + 1} words, {len(vectors)}/{len(words)} dataset words matched", flush=True)

    if local_file:
        path = Path(local_file)
        if path.name != "glove.6B.100d.txt":
            raise ValueError("Please provide the original glove.6B.100d.txt")
        with path.open(encoding="utf-8") as stream:
            consume(stream)
        source["local_file_sha256"] = sha256(path)
    else:
        reader = RangeReader(GLOVE_URL)
        with zipfile.ZipFile(reader) as archive:
            info = archive.getinfo("glove.6B.100d.txt")
            source.update({"archive_size": reader.size, "archive_etag": reader.etag,
                           "member_crc32": info.CRC, "member_size": info.file_size})
            with archive.open(info) as member, io.TextIOWrapper(member, encoding="utf-8") as stream:
                consume(stream)  # Read through EOF so ZIP CRC is checked.
        source["downloaded_bytes"] = reader.downloaded
    source.update({"vectors": vectors, "requested_words": len(words), "matched_words": len(vectors),
                   "oov_words": sorted(words - set(vectors)), "lookup_only_no_fitting": True})
    write_json(destination, source)
    print(f"Saved original GloVe lookup subset to {destination}; {len(vectors)}/{len(words)} words matched")
    return source
