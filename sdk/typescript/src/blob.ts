import { HttpClient } from "./http.js";
import type { BlobListResult, BlobPutResult, ImageTransformOptions } from "./types.js";

/**
 * Object storage (`/v1/blob`) and on-the-fly image transforms (`/v1/image`).
 *
 * Objects are raw bytes, so `get` returns a `Uint8Array` rather than a decoded
 * body. Bucket names and every key segment are validated server-side against
 * `[A-Za-z0-9._-]`; `.`, `..` and path separators inside a segment are
 * rejected, which is what makes nested keys safe.
 */
export class BlobClient {
  constructor(private readonly http: HttpClient) {}

  /** Store an object. The write is atomic (temp file + rename). */
  put(
    bucket: string,
    key: string,
    data: ArrayBuffer | Uint8Array | Blob,
    contentType = "application/octet-stream",
  ): Promise<BlobPutResult> {
    return this.http.putBytes(blobPath(bucket, key), data, contentType);
  }

  /** Fetch an object as raw bytes. */
  get(bucket: string, key: string): Promise<Uint8Array> {
    return this.http.getBytes(blobPath(bucket, key));
  }

  delete(bucket: string, key: string): Promise<{ ok: boolean }> {
    return this.http.delete(blobPath(bucket, key));
  }

  /** List object keys in a bucket. */
  list(bucket: string): Promise<BlobListResult> {
    return this.http.get(`/v1/blob/${encodeURIComponent(bucket)}`);
  }

  /**
   * Resize and/or convert an object already in the blob store, returning the
   * transformed bytes. The stored object is never modified, so this is safe to
   * call repeatedly with different parameters.
   */
  image(bucket: string, key: string, options: ImageTransformOptions = {}): Promise<Uint8Array> {
    return this.http.getBytes(
      `/v1/image/${encodeURIComponent(bucket)}/${encodeKey(key)}`,
      // Undefined entries are dropped by HttpClient.url, so an omitted
      // dimension stays absent instead of arriving as an empty (zero) value.
      {
        w: options.w,
        h: options.h,
        format: options.format,
        quality: options.quality,
      },
    );
  }
}

function blobPath(bucket: string, key: string): string {
  return `/v1/blob/${encodeURIComponent(bucket)}/${encodeKey(key)}`;
}

/**
 * Encode a key while preserving `/` as a segment separator — the server treats
 * a key as multiple path segments and validates each one, so percent-encoding
 * the slashes would turn a nested key into a single illegal segment.
 */
function encodeKey(key: string): string {
  return key.split("/").map(encodeURIComponent).join("/");
}
