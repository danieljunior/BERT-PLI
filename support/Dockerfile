# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as nvidia
ARG CONTAINER_TIMEZONE=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

FROM nvidia as nvidia-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install nano
RUN uv python install 3.9 --default --verbose
WORKDIR /src
COPY . .
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --verbose

ENTRYPOINT ["tail", "-f", "/dev/null"]