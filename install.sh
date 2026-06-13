#!/usr/bin/env bash
# Luma DB — Unix install script
# Downloads the latest release binary from GitHub and installs it to /usr/local/bin.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.sh | bash
#   # or with a specific version:
#   curl -fsSL ... | bash -s -- --version v3.1.0
#   # or to a custom destination:
#   curl -fsSL ... | bash -s -- --dest ~/.local/bin

set -euo pipefail

REPO="Jairodaniel-17/rust-kiss-vdb"
BINARY_NAME="luma"
INSTALL_DIR="/usr/local/bin"
VERSION="latest"

# ----- argument parsing -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --dest)    INSTALL_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ----- detect OS / arch -----
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux*)
    case "$ARCH" in
      x86_64)  TARGET="linux-x86_64-musl" ;;
      aarch64) TARGET="linux-aarch64-gnu" ;;
      *)       echo "Unsupported Linux architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Darwin*)
    case "$ARCH" in
      x86_64)  TARGET="macos-x86_64" ;;
      arm64)   TARGET="macos-aarch64" ;;
      *)       echo "Unsupported macOS architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS. Use install.ps1 on Windows."
    exit 1
    ;;
esac

# ----- resolve version -----
if [[ "$VERSION" == "latest" ]]; then
  echo "Fetching latest release tag..."
  VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
  echo "Latest version: $VERSION"
fi

# ----- download -----
ARCHIVE="${BINARY_NAME}-${VERSION}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ARCHIVE}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL ..."
curl -fsSL "$URL" -o "$TMP/$ARCHIVE"

# ----- verify checksum (if available) -----
SUMS_URL="https://github.com/${REPO}/releases/download/${VERSION}/SHA256SUMS.txt"
if curl -fsSL "$SUMS_URL" -o "$TMP/SHA256SUMS.txt" 2>/dev/null; then
  echo "Verifying checksum..."
  (cd "$TMP" && grep "$ARCHIVE" SHA256SUMS.txt | sha256sum -c -)
  echo "Checksum OK."
fi

# ----- extract and install -----
tar xzf "$TMP/$ARCHIVE" -C "$TMP"
BINARY="$TMP/$BINARY_NAME"
if [[ ! -f "$BINARY" ]]; then
  BINARY="$(find "$TMP" -maxdepth 2 -name "$BINARY_NAME" -type f | head -1)"
fi
chmod +x "$BINARY"

mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/$BINARY_NAME"

echo ""
echo "Luma DB $VERSION installed to $INSTALL_DIR/$BINARY_NAME"
echo ""
echo "Quick start:"
echo "  luma serve              # start server on port 8080"
echo "  LUMA_API_KEY=<secret> luma serve --port 1234"
echo ""
echo "Docs: https://github.com/${REPO}#readme"
