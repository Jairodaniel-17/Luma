FROM debian:trixie-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ cmake make git ca-certificates libfuse3-dev fuse3 python3 procps redis-server postgresql curl >/dev/null
RUN curl -sfL https://github.com/etcd-io/etcd/releases/download/v3.5.17/etcd-v3.5.17-linux-amd64.tar.gz \
    | tar xz -C /usr/local/bin --strip-components=1 etcd-v3.5.17-linux-amd64/etcd
RUN git clone --depth 1 https://github.com/dsrhaslab/lazyfs.git /opt/lazyfs && cd /opt/lazyfs/libs/libpcache && ./build.sh >/dev/null && cd /opt/lazyfs/lazyfs && ./build.sh >/dev/null
