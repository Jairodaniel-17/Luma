#!/bin/bash
# Disaster drill (SPEC-friochap F6) with Docker: MinIO as the remote, one
# container as the host that is lost, a second one as the new host that only
# gets the remote. Repeats DRILLS times.
#
#   scripts/dr_drill.sh <dir with the linux luma binary>
#
# Needs the luma-apagon image (scripts/apagon/Dockerfile) and
# cgr.dev/chainguard/minio.
set -euo pipefail

# `pwd -W`: the Windows path Docker Desktop needs under Git Bash.
BIN=$(cd "$1" && { pwd -W 2>/dev/null || pwd; })
HERE=$(cd "$(dirname "$0")" && { pwd -W 2>/dev/null || pwd; })
DRILLS=${DRILLS:-3}
NET=luma-dr
MINIO=dr-minio
REMOTE=(
  -e BACKUP_REMOTE_URL=s3://luma-dr/friochap
  -e BACKUP_REMOTE_ENDPOINT=http://$MINIO:9000
  -e BACKUP_REMOTE_ALLOW_HTTP=true
  -e BACKUP_REMOTE_REGION=us-east-1
  -e AWS_ACCESS_KEY_ID=drill-access-key
  -e AWS_SECRET_ACCESS_KEY=drill-secret-key-0123456789
  -e WAL_SHIP_INTERVAL_SECS=${WAL_SHIP_INTERVAL_SECS:-10}
)

docker network create $NET >/dev/null 2>&1 || true
failed=0
for drill in $(seq "$DRILLS"); do
  docker rm -f $MINIO >/dev/null 2>&1 || true
  # A fresh, empty remote each time; the bucket is a directory MinIO picks up.
  docker run -d --name $MINIO --network $NET \
    -e MINIO_ROOT_USER=drill-access-key -e MINIO_ROOT_PASSWORD=drill-secret-key-0123456789 \
    --entrypoint sh cgr.dev/chainguard/minio:latest -c "mkdir -p /data/luma-dr && exec minio server /data" >/dev/null
  journal=$(docker volume create)
  run() {
    docker run --rm --network $NET "${REMOTE[@]}" -v "$journal:/var/dr/journal" \
      -v "$BIN:/luma" -v "$HERE/apagon:/apagon" --entrypoint bash luma-apagon /apagon/dr.sh "$1"
  }
  echo "== drill $drill"
  run primary
  run restore || failed=1
  docker volume rm "$journal" >/dev/null
done
docker rm -f $MINIO >/dev/null 2>&1 || true
exit $failed
