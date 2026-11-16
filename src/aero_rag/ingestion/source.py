"""Where manuals are read from. Always S3, so the job behaves the same on a laptop and on ECS."""

from collections.abc import Iterator
from typing import Any


class S3Source:
    def __init__(self, s3_client: Any, bucket: str, prefix: str = ""):
        self._s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix

    def iter_documents(self) -> Iterator[tuple[str, str]]:
        """Yield (key, text) for every .md object under the prefix, in key order."""
        keys = []
        # A paginator follows continuation tokens, so this still works past 1,000 objects.
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            keys += [obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith(".md")]
        for key in sorted(keys):
            body = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            yield key, body.decode("utf-8")
