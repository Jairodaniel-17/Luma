FROM debian:trixie-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ cmake make git ca-certificates libfuse3-dev fuse3 python3 procps redis-server postgresql >/dev/null
RUN git clone --depth 1 https://github.com/dsrhaslab/lazyfs.git /opt/lazyfs && cd /opt/lazyfs/libs/libpcache && ./build.sh >/dev/null && cd /opt/lazyfs/lazyfs && ./build.sh >/dev/null
