#!/usr/bin/env bash
# Resolves a TensorFlow-compatible Python (3.12 or 3.11).
# Source this script, then use $PYTHON_BIN.
#
# Optional env:
#   PYTHON_VERSION=3.12|3.11  (default: try 3.12 then 3.11)
#   INSTALL_PYTHON=1          (default: 1 on bootstrap, set 0 to only resolve)

log() {
  echo "$@" >&2
}

ubuntu_codename() {
  if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${VERSION_CODENAME:-}"
  fi
}

resolve_python_bin() {
  local version="$1"
  if command -v "python${version}" >/dev/null 2>&1; then
    echo "python${version}"
    return 0
  fi
  return 1
}

pyenv_python_path() {
  local patch_version="$1"
  local root="${PYENV_ROOT:-$HOME/.pyenv}"
  echo "${root}/versions/${patch_version}/bin/python"
}

patch_version_for() {
  local version="$1"
  case "$version" in
    3.12) echo "3.12.8" ;;
    3.11) echo "3.11.11" ;;
    *) echo "${version}.8" ;;
  esac
}

find_existing_python() {
  local preferred="${PYTHON_VERSION:-}"

  if [ -n "$preferred" ]; then
    resolve_python_bin "$preferred" && return 0
    return 1
  fi

  resolve_python_bin 3.12 || resolve_python_bin 3.11
}

find_existing_pyenv_python() {
  local version="$1"
  local patch_version
  patch_version="$(patch_version_for "$version")"
  local python_path
  python_path="$(pyenv_python_path "$patch_version")"

  if [ -x "$python_path" ]; then
    echo "$python_path"
    return 0
  fi
  return 1
}

install_python_from_apt() {
  local version="${1:-3.12}"
  local codename
  codename="$(ubuntu_codename)"

  # Ubuntu 26.04 only ships Python 3.14 via apt.
  if [ "$codename" = "resolute" ]; then
    log "==> Skipping apt on Ubuntu 26.04 (python${version} not available)"
    return 1
  fi

  log "==> Installing Python ${version} via apt"
  sudo apt update
  sudo apt install -y software-properties-common

  # deadsnakes is only needed on Ubuntu 22.04
  if [ "$codename" = "jammy" ]; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt update
  fi

  sudo apt install -y \
    "python${version}" \
    "python${version}-venv" \
    "python${version}-dev"
}

install_python_build_deps() {
  sudo apt update
  sudo apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncursesw5-dev \
    libncurses-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libffi-dev \
    liblzma-dev
}

setup_pyenv() {
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"

  if [ -d "$PYENV_ROOT" ] && [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
    log "==> Removing incomplete pyenv installation at $PYENV_ROOT"
    rm -rf "$PYENV_ROOT"
  fi

  if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
    log "==> Installing pyenv"
    curl -fsSL https://pyenv.run | bash
  fi

  export PATH="$PYENV_ROOT/bin:$PATH"
  # shellcheck disable=SC1090
  eval "$(pyenv init -)"
}

install_python_with_pyenv() {
  local version="${1:-3.12}"
  local patch_version
  patch_version="$(patch_version_for "$version")"
  local python_path

  log "==> Installing Python ${patch_version} via pyenv (this may take several minutes)"
  install_python_build_deps
  setup_pyenv

  if ! pyenv install -s "$patch_version"; then
    log "ERROR: pyenv failed to install Python ${patch_version}"
    return 1
  fi

  python_path="$(pyenv_python_path "$patch_version")"
  if [ ! -x "$python_path" ]; then
    log "ERROR: Expected python binary not found at ${python_path}"
    return 1
  fi

  echo "$python_path"
}

verify_python_bin() {
  local bin="$1"
  [ -n "$bin" ] || return 1

  if [ -x "$bin" ]; then
    "$bin" --version >/dev/null 2>&1
    return $?
  fi

  if command -v "$bin" >/dev/null 2>&1; then
    "$bin" --version >/dev/null 2>&1
    return $?
  fi

  return 1
}

ensure_python() {
  local bin=""
  local versions_to_try=()

  if [ -n "${PYTHON_VERSION:-}" ]; then
    versions_to_try=("$PYTHON_VERSION")
  else
    versions_to_try=(3.12 3.11)
  fi

  bin="$(find_existing_python || true)"
  if verify_python_bin "$bin"; then
    PYTHON_BIN="$bin"
    log "==> Using existing $($PYTHON_BIN --version)"
    return 0
  fi

  for version in "${versions_to_try[@]}"; do
    bin="$(find_existing_pyenv_python "$version" || true)"
    if verify_python_bin "$bin"; then
      PYTHON_BIN="$bin"
      log "==> Using existing pyenv $($PYTHON_BIN --version)"
      return 0
    fi
  done

  if [ "${INSTALL_PYTHON:-1}" != "1" ]; then
    log "ERROR: No compatible Python found (need 3.12 or 3.11). Set INSTALL_PYTHON=1 to install."
    return 1
  fi

  for version in "${versions_to_try[@]}"; do
    log "==> Python ${version} not found, attempting apt install"
    if install_python_from_apt "$version"; then
      bin="$(resolve_python_bin "$version" || true)"
      if verify_python_bin "$bin"; then
        PYTHON_BIN="$bin"
        log "==> Installed $($PYTHON_BIN --version) via apt"
        return 0
      fi
    fi
    log "==> apt install unavailable for Python ${version}"
  done

  log "==> Falling back to pyenv"
  bin="$(install_python_with_pyenv "${versions_to_try[0]}")" || return 1

  if ! verify_python_bin "$bin"; then
    log "ERROR: pyenv install did not produce a working Python binary"
    return 1
  fi

  PYTHON_BIN="$bin"
  log "==> Installed $($PYTHON_BIN --version) via pyenv"
}
